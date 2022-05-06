from pathlib import Path

import pandas as pd
import vaex

SAVE_DIR = Path('artifacts/user_features')
SAVE_DIR.mkdir(exist_ok=True, parents=True)


def create_user_ohe_agg(dataset, week):
    transactions = pd.read_pickle(f'./input/{dataset}/transactions_train.pkl')[['user', 'item', 'week']]
    users = pd.read_pickle(f'./input/{dataset}/users.pkl')
    items = pd.read_pickle(f'./input/{dataset}/items.pkl')

    tr = vaex.from_pandas(transactions.query("week >= @week")[['user', 'item']])

    target_columns = [c for c in items.columns if c.endswith('_idx')]
    for c in target_columns:
        save_path = SAVE_DIR / f'user_ohe_agg_dataset{dataset}_week{week}_{c}.pkl'
        tmp = tr.join(vaex.from_pandas(pd.get_dummies(items[['item', c]], columns=[c])), on='item')
        tmp = tmp.drop(columns='item')

        tmp = tmp.groupby('user').agg(['mean'])

        users = vaex.from_pandas(users[['user']]).join(tmp, on='user', how='left').to_pandas_df()
        users = users.rename(columns={
            c: f'user_ohe_agg_{c}' for c in users.columns if c != 'user'
        })

        users = users.sort_values(by='user').reset_index(drop=True)
        users.to_pickle(save_path)
        print("saved", save_path)


if __name__ == '__main__':
    for dataset in ['1', '100']:
        for week in range(8):
            create_user_ohe_agg(dataset, week)
