import gc
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import catboost
import faiss
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import vaex
from tqdm.auto import tqdm

from lfm import calc_embeddings, calc_scores
from metric import apk, mapk
from utils import plot_images


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    print(f'[{name}] {time.time() - start_time:.3f} s')


def objective(trial: optuna.Trial):
    dataset = '100'

    transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
    users = pd.read_pickle(f"input/{dataset}/users.pkl")
    items = pd.read_pickle(f"input/{dataset}/items.pkl")

    class CFG:
        model_type = 'CatBoost'
        popular_num_items = trial.suggest_int('popular_num_items', 12, 64)
        popular_weeks = 1
        train_weeks = 2  # for fast iterations
        item2item_num_items = 24  # the larger the better
        cooc_weeks = trial.suggest_int('cooc_weeks', 4, 50)
        cooc_threshold_ratio = trial.suggest_uniform('cooc_threshold_ratio', 0.5, 2.0)
        ohe_distance_num_items = trial.suggest_int('ohe_distance_num_items', 12, 64)
        ohe_distance_num_weeks = trial.suggest_int('ohe_distance_num_weeks', 1, 100)
        user_transaction_feature_weeks = trial.suggest_int('user_transaction_feature_weeks', 1, 100)
        item_transaction_feature_weeks = trial.suggest_int('item_transaction_feature_weeks', 1, 100)
        item_age_feature_weeks = trial.suggest_int('item_age_feature_weeks', 1, 100)
        user_volume_feature_weeks = trial.suggest_int('user_volume_feature_weeks', 1, 100)
        item_volume_feature_weeks = trial.suggest_int('item_volume_feature_weeks', 1, 100)
        user_item_volume_feature_weeks = trial.suggest_int('user_item_volume_feature_weeks', 1, 100)
        age_volume_feature_weeks = trial.suggest_int('age_volume_feature_weeks', 1, 100)

    def create_candidates(transactions: pd.DataFrame, target_users: np.ndarray, week: int) -> pd.DataFrame:
        """
        transactions
            original transactions (user, item, week)
        target_users, week
            候補生成対象のユーザー
            weekで指定されている週の段階での情報のみから作られる
        """
        print(f"create candidates (week: {week})")
        assert len(target_users) == len(set(target_users))

        def create_candidates_repurchase(
            strategy: str,
            transactions: pd.DataFrame,
            target_users: np.ndarray,
            week_start: int,
            max_items_per_user: int = 1234567890
        ) -> pd.DataFrame:
            tr = transactions.query(
                "user in @target_users and @week_start <= week")[['user', 'item', 'week', 'day']].drop_duplicates(ignore_index=True)

            gr_day = tr.groupby(['user', 'item'])['day'].min().reset_index(name='day')
            gr_week = tr.groupby(['user', 'item'])['week'].min().reset_index(name='week')
            gr_volume = tr.groupby(['user', 'item']).size().reset_index(name='volume')

            gr_day['day_rank'] = gr_day.groupby('user')['day'].rank()
            gr_week['week_rank'] = gr_week.groupby('user')['week'].rank()
            gr_volume['volume_rank'] = gr_volume.groupby('user')['volume'].rank(ascending=False)

            candidates = gr_day.merge(gr_week, on=['user', 'item']).merge(gr_volume, on=['user', 'item'])

            candidates['rank_meta'] = 10**9 * candidates['day_rank'] + candidates['volume_rank']
            candidates['rank_meta'] = candidates.groupby('user')['rank_meta'].rank(method='min')
            # item2itemに使う場合は全件使うと無駄に重くなってしまうので削る
            # dayの小ささ, volumeの大きさの辞書順にソートして上位アイテムのみ残す
            # 全部残したい場合はmax_items_per_userに十分大きな数を指定する
            candidates = candidates.query("rank_meta <= @max_items_per_user").reset_index(drop=True)

            candidates = candidates[['user', 'item', 'week_rank', 'volume_rank', 'rank_meta']].rename(
                columns={'week_rank': f'{strategy}_week_rank', 'volume_rank': f'{strategy}_volume_rank'})

            candidates['strategy'] = strategy
            return candidates.drop_duplicates(ignore_index=True)

        def create_candidates_popular(
            transactions: pd.DataFrame,
            target_users: np.ndarray,
            week_start: int,
            num_weeks: int,
            num_items: int,
        ) -> pd.DataFrame:
            tr = transactions.query(
                "@week_start <= week < @week_start + @num_weeks")[['user', 'item']].drop_duplicates(ignore_index=True)
            popular_items = tr['item'].value_counts().index.values[:num_items]
            popular_items = pd.DataFrame({
                'item': popular_items,
                'rank': range(num_items),
                'crossjoinkey': 1,
            })

            candidates = pd.DataFrame({
                'user': target_users,
                'crossjoinkey': 1,
            })

            candidates = candidates.merge(popular_items, on='crossjoinkey').drop('crossjoinkey', axis=1)
            candidates = candidates.rename(columns={'rank': f'pop_rank'})

            candidates['strategy'] = 'pop'
            return candidates.drop_duplicates(ignore_index=True)

        def create_candidates_cooc(
            transactions: pd.DataFrame,
            base_candidates: pd.DataFrame,
            week_start: int,
            num_weeks: int,
            threshold_ratio: int,
        ) -> pd.DataFrame:
            pair_count_threshold = int(dataset) / 100 * num_weeks / 12 * 50 * threshold_ratio
            week_end = week_start + num_weeks
            tr = transactions.query(
                "@week_start <= week < @week_end")[['user', 'item', 'week']].drop_duplicates(ignore_index=True)
            tr = tr.merge(tr.rename(columns={'item': 'item_with', 'week': 'week_with'}), on='user').query(
                "item != item_with and week <= week_with")[['item', 'item_with']].reset_index(drop=True)
            gr_item_count = tr.groupby('item').size().reset_index(name='item_count')
            gr_pair_count = tr.groupby(['item', 'item_with']).size().reset_index(name='pair_count')
            item2item = gr_pair_count.merge(gr_item_count, on='item')
            item2item['ratio'] = item2item['pair_count'] / item2item['item_count']
            item2item = item2item.query("pair_count > @pair_count_threshold").reset_index(drop=True)

            candidates = base_candidates.merge(item2item, on='item').drop(
                ['item', 'pair_count'], axis=1).rename(columns={'item_with': 'item'})
            base_candidates_columns = [c for c in base_candidates.columns if '_' in c]
            base_candidates_replace = {c: f"cooc_{c}" for c in base_candidates_columns}
            candidates = candidates.rename(columns=base_candidates_replace)
            candidates = candidates.rename(columns={'ratio': 'cooc_ratio', 'item_count': 'cooc_item_count'})

            candidates['strategy'] = 'cooc'
            return candidates.drop_duplicates(ignore_index=True)

        def create_candidates_same_product_code(
            items: pd.DataFrame,
            base_candidates: pd.DataFrame
        ) -> pd.DataFrame:
            item2item = items[['item', 'product_code']].merge(items[['item', 'product_code']].rename(
                {'item': 'item_with'}, axis=1), on='product_code')[['item', 'item_with']].query("item != item_with").reset_index(drop=True)

            candidates = base_candidates.merge(item2item, on='item').drop('item', axis=1).rename(columns={'item_with': 'item'})

            candidates['min_rank_meta'] = candidates.groupby(['user', 'item'])['rank_meta'].transform('min')
            candidates = candidates.query("rank_meta == min_rank_meta").reset_index(drop=True)

            base_candidates_columns = [c for c in base_candidates.columns if '_' in c]
            base_candidates_replace = {c: f"same_product_code_{c}" for c in base_candidates_columns}
            candidates = candidates.rename(columns=base_candidates_replace)

            candidates['strategy'] = 'same_product_code'
            return candidates.drop_duplicates(ignore_index=True)

        def create_candidates_ohe_distance(
            transactions: pd.DataFrame,
            users: pd.DataFrame,
            items: pd.DataFrame,
            target_users: np.ndarray,
            week_start: int,
            num_weeks: int,
            num_items: int,
        ) -> pd.DataFrame:
            users_with_ohe = users[['user']].query("user in @target_users")
            cols = [c for c in items.columns if c.endswith('_idx')]
            for c in cols:
                tmp = pd.read_pickle(f"artifacts/user_features/user_ohe_agg_dataset{dataset}_week{week}_{c}.pkl")
                cs = [c for c in tmp.columns if c.endswith('_mean')]
                users_with_ohe = users_with_ohe.merge(tmp[['user'] + cs], on='user')

            users_with_ohe = users_with_ohe.dropna().reset_index(drop=True)
            limited_users = users_with_ohe['user'].values

            recent_items = transactions.query("@week <= week < @week + @num_weeks")['item'].unique()
            items_with_ohe = pd.get_dummies(items[['item'] + cols], columns=cols)
            items_with_ohe = items_with_ohe.query("item in @recent_items").reset_index(drop=True)
            limited_items = items_with_ohe['item'].values

            item_cols = [c for c in items_with_ohe.columns if c != 'item']
            user_cols = [f'user_ohe_agg_{c}_mean' for c in item_cols]
            users_with_ohe = users_with_ohe[['user'] + user_cols]
            items_with_ohe = items_with_ohe[['item'] + item_cols]

            a_users = users_with_ohe.drop('user', axis=1).values.astype(np.float32)
            a_items = items_with_ohe.drop('item', axis=1).values.astype(np.float32)
            a_users = np.ascontiguousarray(a_users)
            a_items = np.ascontiguousarray(a_items)
            index = faiss.index_factory(a_items.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            index.add(a_items)
            distances, idxs = index.search(a_users, num_items)
            return pd.DataFrame({
                'user': np.repeat(limited_users, num_items),
                'item': limited_items[idxs.flatten()],
                'ohe_distance': distances.flatten(),
                'strategy': 'ohe_distance',
            })

        with timer("repurchase"):
            candidates_repurchase = create_candidates_repurchase('repurchase', transactions, target_users, week)
        with timer("popular"):
            candidates_popular = create_candidates_popular(
                transactions, target_users, week, CFG.popular_weeks, CFG.popular_num_items)
        with timer("item2item"):
            candidates_item2item = create_candidates_repurchase(
                'item2item', transactions, target_users, week, CFG.item2item_num_items)
        with timer("cooccurrence"):
            candidates_cooc = create_candidates_cooc(
                transactions, candidates_item2item, week, CFG.cooc_weeks, CFG.cooc_threshold_ratio)
        with timer("same_product_code"):
            candidates_same_product_code = create_candidates_same_product_code(items, candidates_item2item)
        with timer("ohe distance"):
            candidates_ohe_distance = create_candidates_ohe_distance(
                transactions,
                users,
                items,
                target_users,
                week,
                CFG.ohe_distance_num_weeks,
                CFG.ohe_distance_num_items)

        def drop_common_user_item(candidates_target: pd.DataFrame, candidates_reference: pd.DataFrame) -> pd.DataFrame:
            """
            candidates_targetのうちuser, itemの組がcandidates_referenceにあるものを落とす
            """
            tmp = candidates_reference[['user', 'item']].reset_index(drop=True)
            tmp['flag'] = 1
            candidates = candidates_target.merge(tmp, on=['user', 'item'], how='left')
            return candidates.query("flag != 1").reset_index(drop=True).drop('flag', axis=1)

        candidates_cooc = drop_common_user_item(candidates_cooc, candidates_repurchase)
        candidates_same_product_code = drop_common_user_item(candidates_same_product_code, candidates_repurchase)
        candidates_ohe_distance = drop_common_user_item(candidates_ohe_distance, candidates_repurchase)

        candidates = [
            candidates_repurchase,
            candidates_popular,
            candidates_cooc,
            candidates_same_product_code,
            candidates_ohe_distance,
        ]
        candidates = pd.concat(candidates)

        print(f"volume: {len(candidates)}")
        print(f"duplicates: {len(candidates) / len(candidates[['user', 'item']].drop_duplicates())}")

        volumes = candidates.groupby('strategy').size().reset_index(
            name='volume').sort_values(
            by='volume',
            ascending=False).reset_index(
                drop=True)
        volumes['ratio'] = volumes['volume'] / volumes['volume'].sum()
        print(volumes)

        meta_columns = [c for c in candidates.columns if c.endswith('_meta')]
        return candidates.drop(meta_columns, axis=1)

    # valid: week=0
    # train: week=1..CFG.train_weeks
    def func(week):
        target_users = transactions.query("week == @week")['user'].unique()
        return create_candidates(transactions, target_users, week + 1)
    candidates = joblib.Parallel(n_jobs=-1)(joblib.delayed(func)(week) for week in range(1 + CFG.train_weeks))

    def merge_labels(candidates: pd.DataFrame, week: int) -> pd.DataFrame:
        """
        candidatesに対してweekで指定される週のトランザクションからラベルを付与する
        """
        print(f"merge labels (week: {week})")
        labels = transactions[transactions['week'] == week][['user', 'item']].drop_duplicates(ignore_index=True)
        labels['y'] = 1
        original_positives = len(labels)
        labels = candidates.merge(labels, on=['user', 'item'], how='left')
        labels['y'] = labels['y'].fillna(0)

        remaining_positives_total = labels[['user', 'item', 'y']].drop_duplicates(ignore_index=True)['y'].sum()
        recall = remaining_positives_total / original_positives
        print(f"Recall: {recall}")

        volumes = candidates.groupby('strategy').size().reset_index(name='volume')
        remaining_positives = labels.groupby('strategy')['y'].sum().reset_index()
        remaining_positives = remaining_positives.merge(volumes, on='strategy')
        remaining_positives['recall'] = remaining_positives['y'] / original_positives
        remaining_positives['hit_ratio'] = remaining_positives['y'] / remaining_positives['volume']
        remaining_positives = remaining_positives.sort_values(by='y', ascending=False).reset_index(drop=True)
        print(remaining_positives)

        return labels

    for idx in range(len(candidates)):
        candidates[idx] = merge_labels(candidates[idx], idx)

    def drop_trivial_users(labels):
        """
        LightGBMのxendgcやlambdarankでは正例のみや負例のみのuserは学習に無意味なのと、メトリックの計算がおかしくなるので省く
        """
        bef = len(labels)
        df = labels[labels['user'].isin(labels[['user', 'y']].drop_duplicates().groupby(
            'user').size().reset_index(name='sz').query("sz==2").user)].reset_index(drop=True)
        aft = len(df)
        print(f"drop trivial queries: {bef} -> {aft}")
        return df

    for idx in range(len(candidates)):
        candidates[idx]['week'] = idx

    candidates_valid_all = candidates[0].copy()

    for idx in range(len(candidates)):
        candidates[idx] = drop_trivial_users(candidates[idx])

    # age==25でのアイテムボリューム以上になるような幅を各年齢に対して求める
    tr = transactions[['user', 'item']].merge(users[['user', 'age']], on='user')
    age_volume_threshold = len(tr.query("24 <= age <= 26"))

    age_volumes = {age: len(tr.query("age == @age")) for age in range(16, 100)}

    age_shifts = {}
    for age in range(16, 100):
        for i in range(0, 100):
            low = age - i
            high = age + i
            age_volume = 0
            for j in range(low, high + 1):
                age_volume += age_volumes.get(j, 0)
            if age_volume >= age_volume_threshold:
                age_shifts[age] = i
                break
    print(age_shifts)

    def attach_features(
            transactions: pd.DataFrame,
            users: pd.DataFrame,
            items: pd.DataFrame,
            candidates: pd.DataFrame,
            week: int,
            pretrain_week: int) -> pd.DataFrame:
        """
        user, itemに対して特徴を横付けする
        week: これを含めた以前の情報は使って良い
        """
        print(f"attach features (week: {week})")
        n_original = len(candidates)
        df = candidates.copy()

        with timer("user static fetaures"):
            user_features = ['FN', 'Active', 'age', 'club_member_status_idx', 'fashion_news_frequency_idx']
            df = df.merge(users[['user'] + user_features], on='user')

        with timer("item stacic features"):
            item_features = [c for c in items.columns if c.endswith('idx')]
            df = df.merge(items[['item'] + item_features], on='item')

        with timer("user dynamic features (transactions)"):
            week_end = week + CFG.user_transaction_feature_weeks
            tmp = transactions.query(
                "@week <= week < @week_end").groupby('user')[['price', 'sales_channel_id']].agg(['mean', 'std'])
            tmp.columns = ['user_' + '_'.join(a) for a in tmp.columns.to_flat_index()]
            df = df.merge(tmp, on='user', how='left')

        with timer("item dynamic features (transactions)"):
            week_end = week + CFG.item_transaction_feature_weeks
            tmp = transactions.query(
                "@week <= week < @week_end").groupby('item')[['price', 'sales_channel_id']].agg(['mean', 'std'])
            tmp.columns = ['item_' + '_'.join(a) for a in tmp.columns.to_flat_index()]
            df = df.merge(tmp, on='item', how='left')

        with timer("item dynamic features (user features)"):
            week_end = week + CFG.item_age_feature_weeks
            tmp = transactions.query("@week <= week < @week_end").merge(users[['user', 'age']], on='user')
            tmp = tmp.groupby('item')['age'].agg(['mean', 'std'])
            tmp.columns = [f'age_{a}' for a in tmp.columns.to_flat_index()]
            df = df.merge(tmp, on='item', how='left')

        with timer("item freshness features"):
            tmp = transactions.query("@week <= week").groupby('item')['day'].min().reset_index(name='item_day_min')
            tmp['item_day_min'] -= transactions.query("@week == week")['day'].min()
            df = df.merge(tmp, on='item', how='left')

        with timer("item volume features"):
            week_end = week + CFG.item_volume_feature_weeks
            tmp = transactions.query("@week <= week < @week_end").groupby('item').size().reset_index(name='item_volume')
            df = df.merge(tmp, on='item', how='left')

        with timer("user freshness features"):
            tmp = transactions.query("@week <= week").groupby('user')['day'].min().reset_index(name='user_day_min')
            tmp['user_day_min'] -= transactions.query("@week == week")['day'].min()
            df = df.merge(tmp, on='user', how='left')

        with timer("user volume features"):
            week_end = week + CFG.user_volume_feature_weeks
            tmp = transactions.query("@week <= week < @week_end").groupby('user').size().reset_index(name='user_volume')
            df = df.merge(tmp, on='user', how='left')

        with timer("user-item freshness features"):
            tmp = transactions.query("@week <= week").groupby(['user', 'item']
                                                              )['day'].min().reset_index(name='user_item_day_min')
            tmp['user_item_day_min'] -= transactions.query("@week == week")['day'].min()
            df = df.merge(tmp, on=['item', 'user'], how='left')

        with timer("user-item volume features"):
            week_end = week + CFG.user_item_volume_feature_weeks
            tmp = transactions.query(
                "@week <= week < @week_end").groupby(['user', 'item']).size().reset_index(name='user_item_volume')
            df = df.merge(tmp, on=['user', 'item'], how='left')

        with timer("item age volume features"):
            week_end = week + CFG.age_volume_feature_weeks
            tr = transactions.query("@week <= week < @week_end")[['user', 'item']].merge(users[['user', 'age']], on='user')
            item_age_volumes = []
            for age in range(16, 100):
                low = age - age_shifts[age]
                high = age + age_shifts[age]
                tmp = tr.query("@low <= age <= @high").groupby('item').size().reset_index(name='age_volume')
                tmp['age_volume'] = tmp['age_volume'].rank(ascending=False)
                tmp['age'] = age
                item_age_volumes.append(tmp)
            item_age_volumes = pd.concat(item_age_volumes)
            df = df.merge(item_age_volumes, on=['item', 'age'], how='left')

        with timer("ohe dot products"):
            item_target_cols = [c for c in items.columns if c.endswith('_idx')]

            items_with_ohe = pd.get_dummies(items[['item'] + item_target_cols], columns=item_target_cols)

            users_with_ohe = users[['user']]
            for c in item_target_cols:
                tmp = pd.read_pickle(f"artifacts/user_features/user_ohe_agg_dataset{dataset}_week{week}_{c}.pkl")
                assert tmp['user'].tolist() == users_with_ohe['user'].tolist()
                tmp = tmp[['user'] + [c for c in tmp.columns if c.endswith('_mean')]]
                tmp = tmp.drop('user', axis=1)
                users_with_ohe = pd.concat([users_with_ohe, tmp], axis=1)

            assert items_with_ohe['item'].tolist() == items['item'].tolist()
            assert users_with_ohe['user'].tolist() == users['user'].tolist()

            users_items = df[['user', 'item']].drop_duplicates().reset_index(drop=True)
            n_split = 10
            n_chunk = (len(users_items) + n_split - 1) // n_split
            ohe = []
            for i in range(0, len(users_items), n_chunk):
                users_items_small = users_items.iloc[i:i + n_chunk].reset_index(drop=True)
                users_small = users_items_small['user'].values
                items_small = users_items_small['item'].values

                for item_col in item_target_cols:
                    i_cols = [c for c in items_with_ohe.columns if c.startswith(item_col)]
                    u_cols = [f"user_ohe_agg_{c}_mean" for c in i_cols]
                    users_items_small[f'{item_col}_ohe_score'] = (
                        items_with_ohe[i_cols].values[items_small] *
                        users_with_ohe[u_cols].values[users_small]).sum(
                        axis=1)

                ohe_cols = [f'{col}_ohe_score' for col in item_target_cols]
                users_items_small = users_items_small[['user', 'item'] + ohe_cols]

                ohe.append(users_items_small)
            ohe = pd.concat(ohe)
            df = df.merge(ohe, on=['user', 'item'])

        with timer("lfm features"):
            user_reps, _ = calc_embeddings('i_i', dataset, pretrain_week, 16)
            df = df.merge(user_reps, on='user')

        assert len(df) == n_original
        return df

    dataset_valid_all = attach_features(transactions, users, items, candidates_valid_all, 1, CFG.train_weeks + 1)
    # pretrained modelの学習期間が評価時と提出時で異なるので、candidatesは残しておく
    # datasets = [attach_features(transactions,users,items,candidates[idx],1 +idx,CFG.train_weeks +1) for idx in range(len(candidates))]
    datasets = joblib.Parallel(n_jobs=-1)(joblib.delayed(attach_features)(transactions, users, items,
                                                                          candidates[idx], 1 + idx, CFG.train_weeks + 1) for idx in range(len(candidates)))

    for idx in range(len(datasets)):
        datasets[idx]['query_group'] = datasets[idx]['week'].astype(str) + '_' + datasets[idx]['user'].astype(str)
        datasets[idx] = datasets[idx].sort_values(by='query_group').reset_index(drop=True)

    def concat_train(datasets, begin, num):
        train = pd.concat([datasets[idx] for idx in range(begin, begin + num)])
        return train

    valid = datasets[0]
    train = concat_train(datasets, 1, CFG.train_weeks)

    feature_columns = [c for c in valid.columns if c not in ['y', 'strategy', 'query_group', 'week']]
    print(feature_columns)

    cat_feature_values = [c for c in feature_columns if c.endswith('idx')]
    cat_features = [feature_columns.index(c) for c in cat_feature_values]
    print(cat_feature_values, cat_features)

    train_dataset = catboost.Pool(
        data=train[feature_columns],
        label=train['y'],
        group_id=train['query_group'],
        cat_features=cat_features)
    valid_dataset = catboost.Pool(
        data=valid[feature_columns],
        label=valid['y'],
        group_id=valid['query_group'],
        cat_features=cat_features)

    params = {
        'loss_function': 'YetiRank',
        'use_best_model': True,
        'one_hot_max_size': 300,
        'iterations': 1000,
        'learning_rate': 0.1,
        'verbose': False,
    }
    model = catboost.CatBoost(params)
    model.fit(train_dataset, eval_set=valid_dataset)

    pred = dataset_valid_all[['user', 'item']].reset_index(drop=True)
    pred['pred'] = model.predict(dataset_valid_all[feature_columns])

    pred = pred.groupby(['user', 'item'])['pred'].max().reset_index()
    pred = pred.sort_values(by=['user', 'pred'], ascending=False).reset_index(
        drop=True).groupby('user')['item'].apply(lambda x: list(x)[:12]).reset_index()

    gt = transactions.query("week == 0").groupby('user')['item'].apply(list).reset_index().rename(columns={'item': 'gt'})
    merged = gt.merge(pred, on='user', how='left')
    merged['item'] = merged['item'].fillna('').apply(list)

    return mapk(merged['gt'], merged['item'])


if __name__ == '__main__':
    study = optuna.create_study(
        study_name="dataset_100_0",
        storage="sqlite:///tune/dataset_100_0.db",
        direction='maximize',
        load_if_exists=True,
    )

    study.optimize(objective, timeout=3600 * 48)
