from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def blend(items):
    n_sub = len(items)
    scores = defaultdict(float)
    weights = [1.0, 1.18, 1.19, 0.06]
    for i in range(n_sub):
        for j in range(12):
            scores[items[i][j]] += weights[i] * 1 / (j + 1)
    a = scores.items()
    a = sorted(a, key=lambda x: -x[1])[:12]
    return [b[0] for b in a]


if __name__ == '__main__':
    sub_loc6w = pd.read_csv('submissions/local6.csv')
    sub_loc8w = pd.read_csv('submissions/local8.csv')
    sub_pub8w = pd.read_csv('submissions/public8.csv')
    sub_pub12w = pd.read_csv('submissions/public12w.csv')

    sub = sub_loc6w.copy().rename(columns={'prediction': 'loc6w'})
    sub = sub.merge(sub_loc8w.rename(columns={'prediction': 'loc8w'}))
    sub = sub.merge(sub_pub8w.rename(columns={'prediction': 'pub8w'}))
    sub = sub.merge(sub_pub12w.rename(columns={'prediction': 'pub12w'}))

    predictions = []
    for _, row in tqdm(sub.iterrows()):
        items = [row['loc6w'], row['loc8w'], row['pub12w'], row['pub8w']]
        predictions.append(blend(items))
    sub['prediction'] = predictions
    sub.to_csv("final.csv", index=False)
