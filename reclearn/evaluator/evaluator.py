"""
Created on Nov 14, 2021
Evaluate Functions.
@author: Ziyao Geng(zggzy1996@163.com)
"""
from reclearn.evaluator.metrics import *


def eval_pos_neg(model, test_data, metric_names, k=10, batch_size=None):
    """Evaluate the performance of Top-k recommendation algorithm.
    Note: Test data must contain some negative samples(>= k) and one positive samples.
    Args:
        :param model: A model built-by tensorflow.
        :param test_data: A dict.
        :param metric_names: A list like ['hr'].
        :param k: A scalar(int).
        :param batch_size: A scalar(int).
    :return: A result dict such as {'hr':, 'ndcg':, ...}
    """
    pred_y = - model.predict(test_data, batch_size)
    rank = pred_y.argsort().argsort()[:, 0]
    res_dict = {}
    for name in metric_names:
        if name == 'hr':
            res = hr(rank, k)
        elif name == 'ndcg':
            res = ndcg(rank, k)
        elif name == 'mrr':
            res = mrr(rank, k)
        else:
            break
        res_dict[name] = res
    return res_dict