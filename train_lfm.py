import os
import pickle
from pathlib import Path

import fire
import pandas as pd
from lightfm import LightFM
from scipy import sparse

ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_lfm(dataset: str, train_week: int):
    transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
    n_user = len(pd.read_pickle(f"input/{dataset}/users.pkl"))
    n_item = len(pd.read_pickle(f"input/{dataset}/items.pkl"))
    a = transactions.query("@train_week <= week")[['user', 'item']].drop_duplicates(ignore_index=True)

    a_train = sparse.lil_matrix((n_user, n_item))
    a_train[a['user'], a['item']] = 1

    lightfm_params = {
        'no_components': 16,
        'learning_schedule': 'adadelta',
        'loss': 'bpr',
        'learning_rate': 0.005,
        'random_state': 42,
    }

    model = LightFM(**lightfm_params)
    model.fit(a_train, epochs=100, num_threads=os.cpu_count(), verbose=True)
    save_path = ARTIFACTS_DIR / f"lfm_{dataset}_{train_week}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"saved {save_path}")


if __name__ == '__main__':
    fire.Fire(train_lfm)
