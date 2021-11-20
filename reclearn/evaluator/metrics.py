"""
Created on Nov 14, 2021
@author: Ziyao Geng(zggzy1996@163.com)
"""
import numpy as np


def hr(rank, k):
    """Hit Rate.
    Args:
        :param rank: A list.
        :param k: A scalar(int).
    :return: hit rate.
    """
    res = 0.0
    for r in rank:
        if r < k:
            res += 1
    return res / len(rank)


def mrr(rank, k):
    """Mean Reciprocal Rank.
    Args:
        :param rank: A list.
        :param k: A scalar(int).
    :return: mrr.
    """
    mrr = 0.0
    for r in rank:
        if r < k:
            mrr += 1 / (r + 1)
    return mrr / len(rank)


def ndcg(rank, k):
    """Normalized Discounted Cumulative Gain.
    Args:
        :param rank: A list.
        :param k: A scalar(int).
    :return: ndcg.
    """
    res = 0.0
    for r in rank:
        if r < k:
            res += 1 / np.log2(r + 2)
    return res / len(rank)
