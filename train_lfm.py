import os
import pickle
from pathlib import Path

import fire
import pandas as pd
from lightfm import LightFM
from scipy import sparse

ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)

LIGHTFM_PARAMS = {
    'learning_schedule': 'adadelta',
    'loss': 'bpr',
    'learning_rate': 0.005,
    'random_state': 42,
}
EPOCHS = 100


class Command:
    def i_i(self, dataset: str, week: int, dim: int):
        path_prefix = ARTIFACTS_DIR / f"lfm_i_i_dataset{dataset}_week{week}_dim{dim}"
        print(path_prefix)
        transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
        users = pd.read_pickle(f"input/{dataset}/users.pkl")
        items = pd.read_pickle(f"input/{dataset}/items.pkl")
        n_user = len(users)
        n_item = len(items)
        a = transactions.query("@week <= week")[['user', 'item']].drop_duplicates(ignore_index=True)
        a_train = sparse.lil_matrix((n_user, n_item))
        a_train[a['user'], a['item']] = 1

        lightfm_params = LIGHTFM_PARAMS.copy()
        lightfm_params['no_components'] = dim

        model = LightFM(**lightfm_params)
        model.fit(a_train, epochs=EPOCHS, num_threads=os.cpu_count(), verbose=True)
        save_path = f"{path_prefix}_model.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    def if_i(self, dataset: str, week: int, dim: int):
        path_prefix = ARTIFACTS_DIR / f"lfm_if_i_dataset{dataset}_week{week}_dim{dim}"
        print(path_prefix)
        transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
        users = pd.read_pickle(f"input/{dataset}/users.pkl")
        items = pd.read_pickle(f"input/{dataset}/items.pkl")
        n_user = len(users)
        n_item = len(items)
        a = transactions.query("@week <= week")[['user', 'item']].drop_duplicates(ignore_index=True)
        a_train = sparse.lil_matrix((n_user, n_item))
        a_train[a['user'], a['item']] = 1

        lightfm_params = LIGHTFM_PARAMS.copy()
        lightfm_params['no_components'] = dim

        tmp = users[['age']].reset_index(drop=True)
        tmp['age_nan'] = users['age'].isna()
        tmp['age'] = tmp['age'].fillna(0.0)
        tmp = tmp.astype('float32')
        id = sparse.identity(n_user, dtype='f', format='csr')
        user_features = sparse.hstack([id, tmp.values]).astype('float32')

        model = LightFM(**lightfm_params)
        model.fit(a_train, user_features=user_features, epochs=EPOCHS, num_threads=os.cpu_count(), verbose=True)

        sparse.save_npz(f"{path_prefix}_user_features.npz", user_features)
        with open(f"{path_prefix}_model.pkl", 'wb') as f:
            pickle.dump(model, f)

    def if_f(self, dataset: str, week: int, dim: int):
        path_prefix = ARTIFACTS_DIR / f"lfm_if_f_dataset{dataset}_week{week}_dim{dim}"
        print(path_prefix)
        transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
        users = pd.read_pickle(f"input/{dataset}/users.pkl")
        items = pd.read_pickle(f"input/{dataset}/items.pkl")
        n_user = len(users)
        n_item = len(items)
        a = transactions.query("@week <= week")[['user', 'item']].drop_duplicates(ignore_index=True)
        a_train = sparse.lil_matrix((n_user, n_item))
        a_train[a['user'], a['item']] = 1

        lightfm_params = LIGHTFM_PARAMS.copy()
        lightfm_params['no_components'] = dim

        tmp = users[['age']].reset_index(drop=True)
        tmp['age_nan'] = users['age'].isna()
        tmp['age'] = tmp['age'].fillna(0.0)
        tmp = tmp.astype('float32')
        id = sparse.identity(n_user, dtype='f', format='csr')
        user_features = sparse.hstack([id, tmp.values]).astype('float32')

        cols = [c for c in items.columns if c.endswith('_idx')]
        item_features = sparse.csr_matrix(
            pd.concat([pd.get_dummies(items[c], prefix=c)
                       for c in cols], axis=1).astype('float32')
        )

        model = LightFM(**lightfm_params)
        model.fit(
            a_train,
            user_features=user_features,
            item_features=item_features,
            epochs=EPOCHS,
            num_threads=os.cpu_count(),
            verbose=True
        )

        sparse.save_npz(f"{path_prefix}_user_features.npz", user_features)
        sparse.save_npz(f"{path_prefix}_item_features.npz", item_features)
        with open(f"{path_prefix}_model.pkl", 'wb') as f:
            pickle.dump(model, f)

    def if_if(self, dataset: str, week: int, dim: int):
        path_prefix = ARTIFACTS_DIR / f"lfm_if_if_dataset{dataset}_week{week}_dim{dim}"
        print(path_prefix)
        transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
        users = pd.read_pickle(f"input/{dataset}/users.pkl")
        items = pd.read_pickle(f"input/{dataset}/items.pkl")
        n_user = len(users)
        n_item = len(items)
        a = transactions.query("@week <= week")[['user', 'item']].drop_duplicates(ignore_index=True)
        a_train = sparse.lil_matrix((n_user, n_item))
        a_train[a['user'], a['item']] = 1

        lightfm_params = LIGHTFM_PARAMS.copy()
        lightfm_params['no_components'] = dim

        tmp = users[['age']].reset_index(drop=True)
        tmp['age_nan'] = users['age'].isna()
        tmp['age'] = tmp['age'].fillna(0.0)
        tmp = tmp.astype('float32')
        id = sparse.identity(n_user, dtype='f', format='csr')
        user_features = sparse.hstack([id, tmp.values]).astype('float32')

        cols = [c for c in items.columns if c.endswith('_idx')]
        tmp = pd.concat([pd.get_dummies(items[c], prefix=c) for c in cols], axis=1).astype('float32')
        id = sparse.identity(n_item, dtype='f', format='csr')
        item_features = sparse.hstack([id, tmp.values]).astype('float32')

        model = LightFM(**lightfm_params)
        model.fit(
            a_train,
            user_features=user_features,
            item_features=item_features,
            epochs=EPOCHS,
            num_threads=os.cpu_count(),
            verbose=True
        )

        sparse.save_npz(f"{path_prefix}_user_features.npz", user_features)
        sparse.save_npz(f"{path_prefix}_item_features.npz", item_features)
        with open(f"{path_prefix}_model.pkl", 'wb') as f:
            pickle.dump(model, f)


fire.Fire(Command)
