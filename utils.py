import datetime

import pandas as pd
from logzero import logger


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
