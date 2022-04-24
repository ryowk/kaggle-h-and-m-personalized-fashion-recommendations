from pathlib import Path

import pandas as pd

SAVE_DIR = Path('artifacts/user_features')
SAVE_DIR.mkdir(exist_ok=True, parents=True)


def create_user_ohe_agg(dataset, week):
    transactions = pd.read_pickle(f'./input/{dataset}/transactions_train.pkl')[['user', 'item', 'week']]
    users = pd.read_pickle(f'./input/{dataset}/users.pkl')
    items = pd.read_pickle(f'./input/{dataset}/items.pkl')

    target_columns = [c for c in items.columns if c.endswith('_idx')]
    for c in target_columns:
        save_path = SAVE_DIR / f'user_ohe_agg_dataset{dataset}_week{week}_{c}.pkl'
        tmp = transactions.query("week >= @week")[['user', 'item']]
        tmp = tmp.merge(pd.get_dummies(items[['item', c]], columns=[c]), on='item', validate='many_to_one')
        tmp = tmp.drop('item', axis=1)

        tmp = tmp.groupby('user').agg(['mean', 'std'])
        tmp.columns = ['user_ohe_agg_' + '_'.join(a) for a in tmp.columns.to_flat_index()]
        tmp = tmp.reset_index()
        users = users[['user']].merge(tmp, on='user', how='left', validate='one_to_one')

        users = users.sort_values(by='user').reset_index(drop=True)
        users.to_pickle(save_path)
        print("saved", save_path)


if __name__ == '__main__':
    for dataset in ['1', '10', '100']:
        for week in range(6):
            create_user_ohe_agg(dataset, week)
