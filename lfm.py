import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ARTIFACTS_DIR = Path('artifacts')


def _load_resources(model_type: str, dataset: str, week: int, dim: int):
    path_prefix = ARTIFACTS_DIR / f"lfm_{model_type}_dataset{dataset}_week{week}_dim{dim}"
    model_path = f"{path_prefix}_model.pkl"
    user_features_path = f"{path_prefix}_user_features.npz"
    item_features_path = f"{path_prefix}_item_features.npz"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    user_features = None
    item_features = None
    if model_type == 'i_i':
        pass
    elif model_type == 'if_i':
        user_features = sparse.load_npz(user_features_path)
    elif model_type in ['if_f', 'if_if']:
        user_features = sparse.load_npz(user_features_path)
        item_features = sparse.load_npz(item_features_path)
    else:
        raise NotImplementedError()
    return model, user_features, item_features


def calc_embeddings(model_type: str, dataset: str, week: int, dim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, user_features, item_features = _load_resources(model_type, dataset, week, dim)

    biases, embeddings = model.get_user_representations(user_features)
    n_user = len(biases)
    a = np.hstack([embeddings, biases.reshape(n_user, 1)])
    user_embeddings = pd.DataFrame(a, columns=[f"user_rep_{i}" for i in range(dim + 1)])
    user_embeddings = pd.concat([pd.DataFrame({'user': range(n_user)}), user_embeddings], axis=1)

    biases, embeddings = model.get_item_representations(item_features)
    n_item = len(biases)
    a = np.hstack([embeddings, biases.reshape(n_item, 1)])
    item_embeddings = pd.DataFrame(a, columns=[f"item_rep_{i}" for i in range(dim + 1)])
    item_embeddings = pd.concat([pd.DataFrame({'item': range(n_item)}), item_embeddings], axis=1)
    return user_embeddings, item_embeddings


def calc_scores(model_type: str, dataset: str, week: int, dim: int, user_idxs: np.ndarray, item_idxs: np.ndarray) -> np.ndarray:
    model, user_features, item_features = _load_resources(model_type, dataset, week, dim)
    return model.predict(user_idxs, item_idxs, user_features=user_features, item_features=item_features)
