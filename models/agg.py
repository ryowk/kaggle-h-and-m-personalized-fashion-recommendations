import datetime

import numpy as np
import pandas as pd
from metric import apk, mapk
from utils import train_valid_split

transactions = pd.read_pickle('./input/transformed/transactions_train.pkl')[['t_dat', 'user', 'item']]

transactions_train, transactions_valid = train_valid_split(transactions, datetime.date(2020, 9, 16), 21)
valid = transactions_valid.groupby('user')['item'].apply(list).reset_index()
train = transactions_train.groupby('user')['item'].apply(list).reset_index()

pred = pd.read_csv('./models/output/repurchase.csv')
pred['item'] = pred['item'].apply(lambda x: list(map(int, x.split(' '))))

valid = valid.rename(columns={'item': 'item_valid'})
train = train.rename(columns={'item': 'item_train'})
pred = pred.rename(columns={'item': 'item_pred'})

merged = valid.merge(train, on='user', how='left').merge(pred, on='user')
merged['item_train'] = merged['item_train'].fillna('').apply(list)
merged['train_size'] = merged['item_train'].apply(len)

apks = []
for _, row in merged.iterrows():
    apks.append(apk(row['item_valid'], row['item_pred']))
print('total:', np.mean(apks))
merged['apk'] = apks


def analyze(df):
    np.random.seed(42)
    df_zero = df.query("train_size == 0").reset_index(drop=True)
    df_one = df.query("train_size == 1").reset_index(drop=True)
    df_two = df.query("train_size == 2").reset_index(drop=True)
    zero_size, zero_mean, zero_std = df_zero['apk'].agg(['size', 'mean', 'std'])
    one_size, one_mean, one_std = df_one['apk'].agg(['size', 'mean', 'std'])
    two_size, two_mean, two_std = df_two['apk'].agg(['size', 'mean', 'std'])

    df_lower = pd.DataFrame({
        'q': [0, 1, 2],
        'size': [zero_size, one_size, two_size],
        'mean': [zero_mean, one_mean, two_mean],
        'std': [zero_std, one_std, two_std],
    })

    df_higher = df.query("train_size > 2").reset_index(drop=True)
    df_higher['train_size'] += np.random.uniform(0.0, 1e-3, len(df_higher))
    df_higher['q'] = pd.qcut(df_higher['train_size'], 5)

    df_higher = df_higher.groupby('q')['apk'].agg(['size', 'mean', 'std']).reset_index()

    df = pd.concat([df_lower, df_higher])
    df['ratio'] = df['size'] / df['size'].sum()
    df['score_contrib'] = df['mean'] * df['size'] / df['size'].sum()
    return df


print(analyze(merged))
