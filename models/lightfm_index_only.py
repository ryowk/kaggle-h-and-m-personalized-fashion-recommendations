import datetime

import faiss
import numpy as np
import pandas as pd
import psutil
from lightfm import LightFM
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


no_components = 1024
lightfm_params = {
    'no_components': no_components,
    'learning_schedule': 'adadelta',
    'loss': 'bpr',
    'learning_rate': 0.005,
    'item_alpha': 1e-8,
    'user_alpha': 1e-8,
}

train = sparse.lil_matrix((n_user, n_item))
train[transactions_train.user, transactions_train.item] = 1

model = LightFM(**lightfm_params)
model.fit(train, epochs=100, num_threads=psutil.cpu_count(logical=False), verbose=True)

# naive
logger.info("start naive")
index = faiss.index_factory(no_components, "Flat", faiss.METRIC_INNER_PRODUCT)
index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
index.add(model.item_embeddings)
_, predicted_items = index.search(model.user_embeddings, TOPK)

pred = pd.DataFrame({
    'user': users,
    'item': predicted_items.tolist(),
})

tmp = valid.merge(pred, on='user')
logger.info(f"mapk: {mapk(tmp['item_x'], tmp['item_y'])}")

OUTPUT_PATH_0 = './models/output/lightfm_index_only_0.csv'
pred['item'] = pred['item'].apply(lambda x: ' '.join(map(str, x)))
pred.to_csv(OUTPUT_PATH_0, index=False)
logger.info(f"saved {OUTPUT_PATH_0}")

# refined
logger.info("start refined")
index = faiss.index_factory(no_components, "Flat", faiss.METRIC_INNER_PRODUCT)
index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
index.add(model.item_embeddings)
_, candidate_items = index.search(model.user_embeddings, 100)

high_bias_items = np.argsort(model.item_biases)[::-1][:100]
high_bias_items = np.array([high_bias_items] * n_user)
candidates = np.hstack([candidate_items, high_bias_items])

user_idxs = np.repeat(range(n_user), 200)

result = model.predict(user_idxs, candidates.flatten(), num_threads=psutil.cpu_count(logical=False))
result = result.reshape(n_user, 200)

idxs_each_user = np.argsort(result, axis=1)[:, ::-1][:, :TOPK]
predicted_items = np.array([candidates[i, x] for i, x in enumerate(idxs_each_user)])

pred = pd.DataFrame({
    'user': users,
    'item': predicted_items.tolist(),
})

tmp = valid.merge(pred, on='user')
logger.info(f"mapk: {mapk(tmp['item_x'], tmp['item_y'])}")

OUTPUT_PATH_1 = './models/output/lightfm_index_only_1.csv'
pred['item'] = pred['item'].apply(lambda x: ' '.join(map(str, x)))
pred.to_csv(OUTPUT_PATH_1, index=False)
logger.info(f"saved {OUTPUT_PATH_1}")
