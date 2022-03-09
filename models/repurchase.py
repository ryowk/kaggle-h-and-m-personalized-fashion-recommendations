import datetime

import pandas as pd
from logzero import logger
from metric import mapk
from utils import train_valid_split

transactions = pd.read_pickle('./input/transformed/transactions_train.pkl')[['t_dat', 'user', 'item']]
users = pd.read_pickle('./input/transformed/users.pkl')['user'].values
n_user = len(users)
TOPK = 12

VALID_START_DATE = datetime.date(2020, 9, 16)

transactions_train, transactions_valid = train_valid_split(transactions, VALID_START_DATE, 21)
valid = transactions_valid.groupby('user')['item'].apply(list).reset_index()


def calc_week_pred(week):
    start_date = VALID_START_DATE - datetime.timedelta(days=7 * week)
    end_date = VALID_START_DATE - datetime.timedelta(days=1)
    transactions_week = transactions.query("@start_date <= t_dat <= @end_date")
    week_pred = transactions_week.groupby(['user', 'item']).size().reset_index(name='sz').sort_values(
        by=['user', 'sz'], ascending=False).groupby('user')['item'].apply(list).reset_index()
    week_pred = week_pred.rename(columns={'item': f'item{week}'})
    return week_pred


week_preds = [calc_week_pred(week) for week in range(1, 4)]

start_date = VALID_START_DATE - datetime.timedelta(days=1)
end_date = VALID_START_DATE - datetime.timedelta(days=1)
popular_articles = transactions.query("@start_date <= t_dat <= @end_date").groupby('item').size(
).reset_index(name='sz').sort_values(by='sz', ascending=False)['item'][:TOPK].tolist()

pred = pd.DataFrame({
    'user': users,
    'item': [[] for _ in range(n_user)],
})

for idx, week_pred in enumerate(week_preds):
    week = idx + 1
    pred = pred.merge(week_pred, on='user', how='left')
    pred[f'item{week}'] = pred[f'item{week}'].fillna('').apply(list)
pred['popular_articles'] = [popular_articles] * n_user
for week in range(1, 4):
    pred['item'] += pred[f'item{week}']
pred['item'] += pred['popular_articles']

pred = pred[['user', 'item']]
pred['item'] = pred['item'].apply(lambda x: list(dict.fromkeys(x))[:TOPK])

valid = valid.merge(pred, on='user')
logger.info(f"mapk: {mapk(valid['item_x'], valid['item_y'])}")

OUTPUT_PATH = './models/output/repurchase.csv'
pred['item'] = pred['item'].apply(lambda x: ' '.join(map(str, x)))
pred.to_csv(OUTPUT_PATH, index=False)
logger.info(f"saved {OUTPUT_PATH}")
