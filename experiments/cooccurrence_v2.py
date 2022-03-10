import datetime

import optuna
import pandas as pd
from metric import mapk

transactions = pd.read_pickle(
    'input/transformed/transactions_train.pkl')[['t_dat', 'user', 'item']].query("t_dat > '2020-05-01'").reset_index(drop=True)
TOPK = 12

valid_start_date = datetime.date(2020, 9, 16)
valid_end_date = datetime.date(2020, 9, 22)
transactions_valid = transactions.query("@valid_start_date <= t_dat <= @valid_end_date")
val = transactions_valid.groupby('user')['item'].apply(list).reset_index()


def cooccurrence_v2(num_weeks, cooc_days, cooc_count, cooc_prob, popular_days):
    """
    * 学習期間の終わりからnum_weeks週間までに購入したもの(新しければ新しいほどリストの上位に)
    * 購入したもののなかで、cooc_daysの期間中の出現頻度がcooc_count以上で、共起確率がcooc_probよりデカいもの
    * 直近popular_days日の人気商品
    * v1との違い: 1週目の購入履歴 -> 1週目の購入履歴の共起 -> 2週目の購入履歴 -> ...
    """
    def calc_week_pred(week):
        start_date = valid_start_date - datetime.timedelta(days=7 * week)
        end_date = valid_start_date - datetime.timedelta(days=1)
        transactions_week = transactions.query("@start_date <= t_dat <= @end_date")
        week_pred = transactions_week.groupby(['user', 'item']).size().reset_index(name='sz').sort_values(
            by=['user', 'sz'], ascending=False).groupby('user')['item'].apply(list).reset_index()
        week_pred = week_pred.rename(columns={'item': f'item_{week}'})
        return week_pred

    week_preds = [calc_week_pred(week) for week in range(1, num_weeks + 1)]

    start_date = valid_start_date - datetime.timedelta(days=cooc_days)
    end_date = valid_start_date - datetime.timedelta(days=1)
    cooc = transactions.query("@start_date <= t_dat <= @end_date")[['user', 'item']].drop_duplicates()
    freq = cooc.groupby('item').size().reset_index(name='freq')
    cooc = cooc.merge(cooc.rename(columns={'item': 'item_with'}), on='user')
    cooc = cooc.query("item != item_with")
    cooc = cooc.groupby(['item', 'item_with']).size().reset_index(name='sz')
    cooc = cooc.merge(freq, on='item').reset_index(drop=True)
    cooc['prob'] = cooc['sz'] / cooc['freq']
    cooc = cooc.query("sz >= @cooc_count and prob >= @cooc_prob")
    cooc = cooc.sort_values(by='prob', ascending=False).reset_index(drop=True)

    def func(x):
        purchased = pd.DataFrame({'item': x})
        y = purchased.merge(cooc, on='item').sort_values(by='prob', ascending=False)['item_with'].values.tolist()
        return x + y

    pred = val[['user']]
    for idx, week_pred in enumerate(week_preds):
        week = idx + 1
        pred = pred.merge(week_pred, on='user', how='left')
        pred[f'item_{week}'] = pred[f'item_{week}'].fillna('').apply(list)
        pred[f'item_{week}'] = pred[f'item_{week}'].apply(func)

    pred['item'] = pred['item_1']
    for week in range(2, num_weeks + 1):
        pred['item'] += pred[f'item_{week}']

    start_date = valid_start_date - datetime.timedelta(days=popular_days)
    end_date = valid_start_date - datetime.timedelta(days=1)
    popular_articles = transactions.query("@start_date <= t_dat <= @end_date").groupby(
        'item').size().reset_index(name='sz').sort_values(by='sz', ascending=False)['item'][:TOPK].tolist()

    pred['popular_articles'] = [popular_articles] * len(pred)
    pred['item'] += pred['popular_articles']
    pred = pred[['user', 'item']].reset_index(drop=True)
    pred['item'] = pred['item'].apply(lambda x: list(dict.fromkeys(x))[:TOPK])
    return mapk(val.item, pred['item'])


def objective(trial):
    num_weeks = trial.suggest_int('num_weeks', 1, 8)
    cooc_days = trial.suggest_int('cooc_days', 1, 120)
    cooc_count = trial.suggest_int('cooc_count', 5, 100)
    cooc_prob = trial.suggest_uniform('cooc_prob', 0.0, 1.0)
    popular_days = trial.suggest_int('popular_days', 1, 28)
    return cooccurrence_v2(num_weeks, cooc_days, cooc_count, cooc_prob, popular_days)


study = optuna.create_study(
    direction='maximize',
    study_name='cooccurrence_v2',
    storage='sqlite:///output/cooccurrence_v2.db',
    load_if_exists=True)
study.optimize(objective, timeout=7200)
study.trials_dataframe().sort_values(
    by='value',
    ascending=False).reset_index(
        drop=True).to_csv(
            'output/cooccurrence_v2.csv',
    index=False)


"""
number,value,datetime_start,datetime_complete,duration,params_cooc_count,params_cooc_days,params_cooc_prob,params_num_weeks,params_popular_days,state
320,0.026131522473255345,2022-03-10 11:54:47.313529,2022-03-10 12:16:39.660893,0 days 00:21:52.347364,8,80,0.05392084069987066,5,1,COMPLETE
345,0.026123999284557987,2022-03-10 12:01:58.731986,2022-03-10 12:25:53.647136,0 days 00:23:54.915150,7,84,0.05478430034479041,5,1,COMPLETE
350,0.026021882715798545,2022-03-10 12:03:05.958399,2022-03-10 12:19:10.525202,0 days 00:16:04.566803,11,84,0.06528758224573698,5,1,COMPLETE
313,0.025801905273974413,2022-03-10 11:51:58.479885,2022-03-10 12:01:07.569626,0 days 00:09:09.089741,34,81,0.04813788836860107,5,1,COMPLETE
324,0.025801483551340255,2022-03-10 11:55:40.983832,2022-03-10 12:04:27.477227,0 days 00:08:46.493395,34,81,0.05497214890932222,5,1,COMPLETE
325,0.025800651327340995,2022-03-10 11:55:43.802968,2022-03-10 12:04:52.656293,0 days 00:09:08.853325,34,81,0.052837355623152504,5,1,COMPLETE
331,0.025798189417623938,2022-03-10 11:58:44.808645,2022-03-10 12:07:46.556034,0 days 00:09:01.747389,35,82,0.05285433297426469,5,1,COMPLETE
321,0.025797750877895487,2022-03-10 11:54:49.582110,2022-03-10 12:03:18.927103,0 days 00:08:29.344993,35,80,0.05318428737424637,5,1,COMPLETE
323,0.025797749221271333,2022-03-10 11:55:37.395418,2022-03-10 12:04:41.826023,0 days 00:09:04.430605,34,81,0.05159197286537275,5,1,COMPLETE
318,0.025797017571127713,2022-03-10 11:54:07.782614,2022-03-10 12:03:05.926880,0 days 00:08:58.144266,34,82,0.05156976603833945,5,1,COMPLETE
"""
