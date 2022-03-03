# https://www.kaggle.com/george86/calculate-map-12-fast-faster-fastest#V3-Implementation!
import numpy as np


def map_at_k(y_true, y_pred, k=12):

    # compute the Rel@K for all items
    rel_at_k = np.zeros((len(y_true), k), dtype=int)

    # collect the intersection indexes (for the ranking vector) for all pairs
    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        _, _, inter_idxs = np.intersect1d(truth, pred[:k], assume_unique=True, return_indices=True)
        rel_at_k[idx, inter_idxs] = 1

    # Calculate the intersection counts for all pairs
    intersection_count_at_k = rel_at_k.cumsum(axis=1)

    # we have the same denominator for all ranking vectors
    ranks = np.arange(1, k + 1, 1)

    # Calculating the Precision@K for all Ks for all pairs
    precisions_at_k = intersection_count_at_k / ranks
    # Multiply with the Rel@K for all pairs
    precisions_at_k = precisions_at_k * rel_at_k

    # Calculate the average precisions @ K for all pairs
    average_precisions_at_k = precisions_at_k.mean(axis=1)

    # calculate the final MAP@K
    map_at_k = average_precisions_at_k.mean()

    return map_at_k
