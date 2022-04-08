import argparse

import pandas as pd
from tqdm import tqdm


def blend(items: list[list[str]]):
    """
    各推薦の上位から順に取ったものを作る

    例えばa, b, cの3つのアイテムリストがある場合
    a[0], b[0], c[0], a[1], b[1], c[1], ...
    という順に並べて重複をなくした上位12件を残す
    """
    n_sub = len(items)
    c = []
    for i in range(12):
        for j in range(n_sub):
            c.append(items[j][i])
    return list(dict.fromkeys(c))[:12]


def blend_submissions(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    複数の提出データフレームを混ぜる
    """
    n_sub = len(dfs)
    dfs = [df.rename(columns={'prediction': f'prediction_{i}'}) for i, df in enumerate(dfs)]
    sub = dfs[0]
    for i in range(1, n_sub):
        sub = sub.merge(dfs[i], on='customer_id', validate='one_to_one')

    for i in range(n_sub):
        sub[f'prediction_{i}'] = sub[f'prediction_{i}'].apply(lambda x: x.split(' '))

    predictions = []
    for _, row in tqdm(sub.iterrows()):
        items = [row[f'prediction_{i}'] for i in range(n_sub)]
        predictions.append(blend(items))
    sub['prediction'] = predictions

    sub = sub[['customer_id', 'prediction']].reset_index(drop=True)
    sub['prediction'] = sub['prediction'].apply(lambda x: ' '.join(x))
    return sub


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=lambda x: x.split(','),
        help="comma separated submission file paths (first: strongest, last: weakest)")
    parser.add_argument("output", type=str, help="output")
    args = parser.parse_args()

    dfs = [pd.read_csv(path) for path in args.paths]
    sub = blend_submissions(dfs)
    sub.to_csv(args.output, index=False)
