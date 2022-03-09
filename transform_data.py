"""
- article_id, category_idを含めた全てのカテゴリを0-indexedな連番に変換する(_idxがついたカラムが追加される)
- None, 1のみのカテゴリを0, 1に変換する(カラムは上書きされる)
- 1, 2のみのカテゴリを0, 1に変換する(カラムは上書きされる)
"""
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from logzero import logger
from tqdm import tqdm

import schema

INPUT_DIR = Path('./input/h-and-m-personalized-fashion-recommendations')
OUTPUT_DIR = Path('./input/transformed')


def _count_encoding_dict(df: pd.DataFrame, col_name: str) -> dict[Any, int]:
    v = df.groupby(col_name).size().reset_index(name='size').sort_values(by='size', ascending=False)[col_name].tolist()
    return {x: i for i, x in enumerate(v)}


def _dict_to_dataframe(mp: dict[Any, int]) -> pd.DataFrame:
    return pd.DataFrame(mp.items(), columns=['val', 'idx'])


def _add_idx_column(df: pd.DataFrame, col_name_from: str, col_name_to: str, mp: dict[Any, int]):
    df[col_name_to] = df[col_name_from].apply(lambda x: mp[x]).astype('int64')


logger.info("start reading dataframes")
articles = pd.read_csv(INPUT_DIR / 'articles.csv', dtype=schema.ARTICLES_ORIGINAL)
customers = pd.read_csv(INPUT_DIR / 'customers.csv', dtype=schema.CUSTOMERS_ORIGINAL)
transactions = pd.read_csv(INPUT_DIR / 'transactions_train.csv', dtype=schema.TRANSACTIONS_ORIGINAL, parse_dates=['t_dat'])
sample_submission = pd.read_csv(
    INPUT_DIR / 'sample_submission.csv',
    usecols=['customer_id'],
    dtype=schema.SAMPLE_SUBMISSION_ORIGINAL)

# customer_id
logger.info("start processing customer_id")
customer_ids = sorted(customers.customer_id.unique())
mp_customer_id = {x: i for i, x in enumerate(customer_ids)}
_dict_to_dataframe(mp_customer_id).to_pickle(OUTPUT_DIR / 'mp_customer_id.pkl')

# article_id
logger.info("start processing article_id")
article_ids = sorted(articles.article_id.unique())
mp_article_id = {x: i for i, x in enumerate(article_ids)}
_dict_to_dataframe(mp_article_id).to_pickle(OUTPUT_DIR / 'mp_article_id.pkl')

################
# customers
################
logger.info("start processing customers")
_add_idx_column(customers, 'customer_id', 'user', mp_customer_id)
# (None, 1) -> (0, 1)
customers['FN'] = customers['FN'].fillna(0).astype('int64')
customers['Active'] = customers['Active'].fillna(0).astype('int64')

# 頻度順に番号を振る(既にintなものも連番のほうが都合が良いので振り直す)
customers['club_member_status'] = customers['club_member_status'].fillna('NULL')
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NULL')
count_encoding_columns = [
    'club_member_status',
    'fashion_news_frequency',
]
for col_name in count_encoding_columns:
    mp = _count_encoding_dict(customers, col_name)
    _add_idx_column(customers, col_name, f'{col_name}_idx', mp)
customers.to_pickle(OUTPUT_DIR / 'users.pkl')

################
# articles
################
logger.info("start processing articles")
_add_idx_column(articles, 'article_id', 'item', mp_article_id)
count_encoding_columns = [
    'product_type_no',
    'product_group_name',
    'graphical_appearance_no',
    'colour_group_code',
    'perceived_colour_value_id',
    'perceived_colour_master_id',
    'department_no',
    'index_code',
    'index_group_no',
    'section_no',
    'garment_group_no',
]
for col_name in count_encoding_columns:
    mp = _count_encoding_dict(articles, col_name)
    _add_idx_column(articles, col_name, f'{col_name}_idx', mp)
articles.to_pickle(OUTPUT_DIR / 'items.pkl')

################
# transactions
################
logger.info("start processing transactions")
_add_idx_column(transactions, 'customer_id', 'user', mp_customer_id)
_add_idx_column(transactions, 'article_id', 'item', mp_article_id)
# (1, 2) -> (0, 1)
transactions['sales_channel_id'] = transactions['sales_channel_id'] - 1
transactions.to_pickle(OUTPUT_DIR / 'transactions_train.pkl')

################
# sample_submission
################
logger.info("start processing sample_submission")
_add_idx_column(sample_submission, 'customer_id', 'user', mp_customer_id)
sample_submission.to_pickle(OUTPUT_DIR / 'sample_submission.pkl')


################
# images
################
INPUT_DIR = Path('./input/h-and-m-personalized-fashion-recommendations/images')
OUTPUT_DIR = Path('./input/transformed/images')
mp = pd.read_pickle('./input/transformed/mp_article_id.pkl')
dct = dict(zip(mp.val, mp.idx))

paths = list(INPUT_DIR.glob("*/*.jpg"))
for path in tqdm(paths):
    name = path.name[:-4]
    idx = dct[name]
    path_to = OUTPUT_DIR / f'{idx}.jpg'
    shutil.copyfile(path, path_to)
