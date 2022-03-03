"""
customer_id_idx, prediction_idxを持ったCSVからcustomer_id, predictionを持ったCSVに変換する

before
    customer_id_idx: int
    prediction_idx: str (space separated 12 int)

after
    customer_id: str
    prediction: str (space separated 12 str)
"""
import fire
import pandas as pd

MP_ARTICLE_CSV = './input/transformed/mp_article_id.csv'
MP_CUSTOMER_CSV = './input/transformed/mp_customer_id.csv'


def transform_submission(input_path: str, output_path: str):
    articles = pd.read_csv(MP_ARTICLE_CSV)['val'].tolist()
    customers = pd.read_csv(MP_CUSTOMER_CSV)['val'].tolist()

    df = pd.read_csv(input_path)[['customer_id_idx', 'prediction_idx']]

    df['customer_id'] = df.customer_id_idx.apply(lambda x: customers[x])

    def _trans_article(x: str) -> str:
        tmp = map(int, x.split(' '))
        return ' '.join(map(lambda x: articles[x], tmp))
    df['prediction'] = df.prediction_idx.apply(_trans_article)
    df[['customer_id', 'prediction']].to_csv(output_path, index=False)


if __name__ == '__main__':
    fire.Fire(transform_submission)
