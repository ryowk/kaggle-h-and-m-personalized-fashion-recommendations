import datetime

import matplotlib.pyplot as plt
import pandas as pd
from logzero import logger
from PIL import Image


def extract_transactions_valid(transactions: pd.DataFrame, valid_start_date: datetime.date) -> pd.DataFrame:
    """
    valid_start_dateからの1週間分を抜き出す
    """
    valid_end_date = valid_start_date + datetime.timedelta(days=7)
    start_date = valid_start_date.strftime("%Y-%m-%d")
    end_date = valid_end_date.strftime("%Y-%m-%d")
    logger.info(f"valid: [{start_date}, {end_date})")
    df = transactions[(start_date <= transactions.t_dat) & (transactions.t_dat < end_date)].reset_index(drop=True)
    logger.info(f"# of records: {len(df)}")
    return df


def extract_transactions_train(transactions: pd.DataFrame, valid_start_date: datetime.date, days: int) -> pd.DataFrame:
    """
    valid_start_date以前のdays日分を抜き出す
    """
    train_start_date = valid_start_date - datetime.timedelta(days=days)
    start_date = train_start_date.strftime("%Y-%m-%d")
    end_date = valid_start_date.strftime("%Y-%m-%d")
    logger.info(f"train: [{start_date}, {end_date})")
    df = transactions[(start_date <= transactions.t_dat) & (transactions.t_dat < end_date)].reset_index(drop=True)
    logger.info(f"# of records: {len(df)}")
    return df


def train_valid_split(transactions: pd.DataFrame, valid_start_date: datetime.date,
                      train_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions_train = extract_transactions_train(transactions, valid_start_date, train_days)
    transactions_valid = extract_transactions_valid(transactions, valid_start_date)
    return transactions_train, transactions_valid


def plot_images(idxs: list[int]):
    paths = [f'./input/transformed/images/{idx}.jpg' for idx in idxs]
    columns = 12
    n = len(idxs)
    rows = (n + columns - 1) // columns
    plt.figure(figsize=(4 * columns, 4 * rows))
    for i, path in enumerate(paths):
        try:
            img = Image.open(path)
        except FileNotFoundError:
            img = Image.open('./input/transformed/images/notfound.png')
        img = img.resize((256, 256))
        plt.subplot(rows, columns, i + 1)

        plt.axis('off')
        plt.imshow(img)
    plt.show()
