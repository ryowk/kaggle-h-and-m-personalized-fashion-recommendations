import datetime

import pandas as pd
from logzero import logger
from metric import mapk
from utils import train_valid_split

transactions = pd.read_pickle('./input/transformed/transactions_train.pkl')[['t_dat', 'user', 'item']]
users = pd.read_pickle('./input/transformed/users.pkl')[['user', 'age']]
users['q_age'] = pd.qcut(users['age'], 5).astype(str)
n_user = len(users)
TOPK = 12

transactions = transactions.merge(users[['user', 'q_age']], on='user')

VALID_START_DATE = datetime.date(2020, 9, 16)

transactions_train, transactions_valid = train_valid_split(transactions, VALID_START_DATE, 21)
valid = transactions_valid.groupby('user')['item'].apply(list).reset_index()

start_date = VALID_START_DATE - datetime.timedelta(days=7)
end_date = VALID_START_DATE - datetime.timedelta(days=1)

results = []
for q_age in sorted(users.q_age.unique()):
    popular_items = transactions.query("@start_date <= t_dat <= @end_date").query("q_age == @q_age").groupby(
        'item').size().reset_index(name='sz').sort_values(by='sz', ascending=False)['item'][:TOPK].tolist()
    results.append({
        'q_age': q_age,
        'item': popular_items,
    })
df = pd.DataFrame(results)
users = users.merge(df, on='q_age')

pred = users[['user', 'item']].reset_index(drop=True)

valid = valid.merge(pred, on='user')
logger.info(f"mapk: {mapk(valid['item_x'], valid['item_y'])}")

OUTPUT_PATH = './models/output/popular_age_dep.csv'
pred['item'] = pred['item'].apply(lambda x: ' '.join(map(str, x)))
pred.to_csv(OUTPUT_PATH, index=False)
logger.info(f"saved {OUTPUT_PATH}")
