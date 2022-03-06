import datetime

import optuna
import pandas as pd
import schema
from metric import mapk

transactions = pd.read_csv(
    'input/transformed/transactions_train.csv',
    parse_dates=['t_dat'],
    usecols=list(
        schema.TRANSACTIONS.keys()) +
    ['t_dat'],
    dtype=schema.TRANSACTIONS)
TOPK = 12


def repurchase(num_weeks, block_size, popular_days):
    """
    そのユーザーが直近1週間に購入したものを購入数順に
    そのユーザーが直近2週間に購入したものを購入数順に
    そのユーザーが直近num_weeks週間に購入したものを購入数順に
    全ユーザーで直近num_weeks週間に購入された上位
    """
    valid_start_date = datetime.date(2020, 9, 16)
    valid_end_date = datetime.date(2020, 9, 22)
    transactions_valid = transactions.query("@valid_start_date <= t_dat <= @valid_end_date")
    val = transactions_valid.groupby('customer_id_idx')['article_id_idx'].apply(list).reset_index()

    def calc_week_pred(week):
        start_date = valid_start_date - datetime.timedelta(days=block_size * week)
        end_date = valid_start_date - datetime.timedelta(days=1)
        transactions_week = transactions.query("@start_date <= t_dat <= @end_date")
        week_pred = transactions_week.groupby(['customer_id_idx', 'article_id_idx']).size().reset_index(name='sz').sort_values(
            by=['customer_id_idx', 'sz'], ascending=False).groupby('customer_id_idx')['article_id_idx'].apply(list).reset_index()
        week_pred = week_pred.rename(columns={'article_id_idx': f'article_id_idx_{week}'})
        return week_pred
    week_preds = [calc_week_pred(week) for week in range(1, num_weeks + 1)]

    start_date = valid_start_date - datetime.timedelta(days=popular_days)
    end_date = valid_start_date - datetime.timedelta(days=1)
    popular_articles = transactions.query("@start_date <= t_dat <= @end_date").groupby('article_id_idx').size(
    ).reset_index(name='sz').sort_values(by='sz', ascending=False)['article_id_idx'][:TOPK].tolist()

    pred = val[['customer_id_idx']]
    for idx, week_pred in enumerate(week_preds):
        week = idx + 1
        pred = pred.merge(week_pred, on='customer_id_idx', how='left')
        pred[f'article_id_idx_{week}'] = pred[f'article_id_idx_{week}'].fillna('').apply(list)
    pred['popular_articles'] = [popular_articles] * len(pred)
    pred['article_id_idx'] = pred['article_id_idx_1']
    for week in range(2, num_weeks + 1):
        pred['article_id_idx'] += pred[f'article_id_idx_{week}']
    pred['article_id_idx'] += pred['popular_articles']
    pred = pred[['customer_id_idx', 'article_id_idx']]
    pred['article_id_idx'] = pred['article_id_idx'].apply(lambda x: list(dict.fromkeys(x))[:TOPK])
    return mapk(val.article_id_idx, pred['article_id_idx'])


def objective(trial):
    num_weeks = trial.suggest_int('num_weeks', 1, 10)
    block_size = trial.suggest_int('block_size', 1, 14)
    popular_days = trial.suggest_int('popular_days', 1, 14)
    return repurchase(num_weeks, block_size, popular_days)


study = optuna.create_study(
    direction='maximize',
    study_name='repurchase',
    storage='sqlite:///output/repurchase.db',
    load_if_exists=True)
study.optimize(objective, n_trials=1)
study.trials_dataframe().sort_values(
    by='value',
    ascending=False).reset_index(
        drop=True).to_csv(
            'output/repurchase.csv',
    index=False)

"""
number,value,datetime_start,datetime_complete,duration,params_block_size,params_num_weeks,params_popular_days,state
717,0.024717037489228325,2022-03-05 22:45:49.272971,2022-03-05 22:50:11.641672,0 days 00:04:22.368701,3,10,1,COMPLETE
409,0.02470405220128637,2022-03-05 22:35:02.429877,2022-03-05 22:36:02.058829,0 days 00:00:59.628952,3,9,1,COMPLETE
616,0.02470405220128637,2022-03-05 22:41:21.025190,2022-03-05 22:42:06.944091,0 days 00:00:45.918901,3,9,1,COMPLETE
570,0.02460209001675657,2022-03-05 22:39:59.521112,2022-03-05 22:40:51.067283,0 days 00:00:51.546171,4,9,1,COMPLETE
327,0.024457537906210268,2022-03-05 22:32:48.115184,2022-03-05 22:33:44.798353,0 days 00:00:56.683169,3,9,5,COMPLETE
539,0.024383577438309365,2022-03-05 22:39:06.661469,2022-03-05 22:40:02.218066,0 days 00:00:55.556597,5,9,1,COMPLETE
265,0.024333205847987784,2022-03-05 22:31:08.581051,2022-03-05 22:32:06.261936,0 days 00:00:57.680885,3,9,9,COMPLETE
257,0.024274786440739,2022-03-05 22:31:00.148683,2022-03-05 22:32:08.019121,0 days 00:01:07.870438,3,10,4,COMPLETE
387,0.024274786440739,2022-03-05 22:34:16.995523,2022-03-05 22:35:04.698460,0 days 00:00:47.702937,3,10,4,COMPLETE
668,0.024253628307239827,2022-03-05 22:42:57.443188,2022-03-05 22:43:41.523359,0 days 00:00:44.080171,3,10,2,COMPLETE
"""
