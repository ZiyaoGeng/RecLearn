import numpy as np


def hr(rank, k):
    res = 0.0
    for r in rank:
        if r < k:
            res += 1
    return res / len(rank)


def mrr(rank, k):
    return


def ndcg(rank, k):
    res = 0.0
    for r in rank:
        if r < k:
            res += 1 / np.log2(r + 2)
    return res / len(rank)
