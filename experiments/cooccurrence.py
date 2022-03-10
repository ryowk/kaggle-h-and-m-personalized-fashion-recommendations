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


def cooccurrence(num_weeks, cooc_days, cooc_count, cooc_prob, popular_days):
    """
    * 学習期間の終わりからnum_weeks週間までに購入したもの(新しければ新しいほどリストの上位に)
    * 購入したもののなかで、cooc_daysの期間中の出現頻度がcooc_count以上で、共起確率がcooc_probよりデカいもの
    * 直近popular_days日の人気商品
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

    pred = val[['user']]
    for idx, week_pred in enumerate(week_preds):
        week = idx + 1
        pred = pred.merge(week_pred, on='user', how='left')
        pred[f'item_{week}'] = pred[f'item_{week}'].fillna('').apply(list)
    pred['item'] = pred['item_1']
    for week in range(2, num_weeks + 1):
        pred['item'] += pred[f'item_{week}']

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

    pred['item'] = pred['item'].apply(lambda x: func(x))

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
    return cooccurrence(num_weeks, cooc_days, cooc_count, cooc_prob, popular_days)


study = optuna.create_study(
    direction='maximize',
    study_name='cooccurrence',
    storage='sqlite:///output/cooccurrence.db',
    load_if_exists=True)
study.optimize(objective, timeout=3600)
study.trials_dataframe().sort_values(
    by='value',
    ascending=False).reset_index(
        drop=True).to_csv(
            'output/cooccurrence.csv',
    index=False)
