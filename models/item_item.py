import datetime
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd
from logzero import logger
from metric import mapk
from scipy import sparse
from utils import train_valid_split

transactions = pd.read_pickle('./input/transformed/transactions_train.pkl')[['t_dat', 'user', 'item']]
users = pd.read_pickle('./input/transformed/users.pkl')['user'].values
items = pd.read_pickle('./input/transformed/items.pkl')['item'].values
n_user = len(users)
n_item = len(items)
TOPK = 12

VALID_START_DATE = datetime.date(2020, 9, 16)

transactions_train, transactions_valid = train_valid_split(transactions, VALID_START_DATE, 21)
valid = transactions_valid.groupby('user')['item'].apply(list).reset_index()

a = transactions_train[['user', 'item']].drop_duplicates()
b = a.groupby('item').size().reset_index(name='sz')
c = a.merge(b, on='item')
c['sz'] = np.sqrt(c['sz'])

a_mat = sparse.lil_matrix((n_user, n_item))
a_mat[c['user'], c['item']] = 1
t_mat = sparse.lil_matrix((n_user, n_item))
t_mat[c['user'], c['item']] = 1.0 / c['sz']

s_mat = t_mat.T.dot(t_mat)
mat = a_mat.dot(s_mat)

s_norm = np.array(np.sum(s_mat, axis=0)).flatten()
s_norm[s_norm == 0.0] = np.inf


def func(u):
    a = np.array(mat[u].todense()).flatten() / s_norm
    return np.argsort(a)[::-1][:12].astype('int32')


with Pool(64) as p:
    items = p.map(func, [u for u in range(n_user)])

pred = pd.DataFrame({
    'user': users,
    'item': items
})

valid = valid.merge(pred, on='user')
logger.info(f"mapk: {mapk(valid['item_x'], valid['item_y'])}")

OUTPUT_PATH = './models/output/item_item.csv'
pred['item'] = pred['item'].apply(lambda x: ' '.join(map(str, x)))
pred.to_csv(OUTPUT_PATH, index=False)
logger.info(f"saved {OUTPUT_PATH}")
