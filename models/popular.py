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

start_date = VALID_START_DATE - datetime.timedelta(days=1)
end_date = VALID_START_DATE - datetime.timedelta(days=1)
popular_articles = transactions.query("@start_date <= t_dat <= @end_date").groupby('item').size(
).reset_index(name='sz').sort_values(by='sz', ascending=False)['item'][:TOPK].tolist()

pred = pd.DataFrame({
    'user': users,
    'item': [popular_articles for _ in range(n_user)],
})

valid = valid.merge(pred, on='user')
logger.info(f"mapk: {mapk(valid['item_x'], valid['item_y'])}")

OUTPUT_PATH = './models/output/popular.csv'
pred['item'] = pred['item'].apply(lambda x: ' '.join(map(str, x)))
pred.to_csv(OUTPUT_PATH, index=False)
logger.info(f"saved {OUTPUT_PATH}")
