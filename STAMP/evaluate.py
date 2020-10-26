'''
Descripttion: Evaluate
Author: Ziyao Geng
Date: 2020-10-25 10:07:17
LastEditors: ZiyaoGeng
LastEditTime: 2020-10-26 09:57:47
'''

import numpy as np


def getHit(pred_y, true_y):
    """
    calculate hit rate
    :return:
    """
    # reversed
    pred_index = np.argsort(-pred_y)[:, :_K]
    return sum([true_y[i] in pred_index[i] for i in range(len(pred_index))]) / len(pred_index)


def getMRR(pred_y, true_y):
    """
    """
    pred_index = np.argsort(-pred_y)[:, :_K]
    return sum([1 / (np.where(true_y[i] == pred_index[i])[0][0] + 1) \
        for i in range(len(pred_index)) if len(np.where(true_y[i] == pred_index[i])[0]) != 0]) / len(pred_index)


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, mrr
    """
    global _K
    _K = K
    test_X, test_y = test
    pred_y = model.predict(test_X)
    hit_rate = getHit(pred_y, test_y)
    mrr = getMRR(pred_y, test_y)
    
    
    return hit_rate, mrr
 
# K = 10
# a = np.random.randint(20, size=(400))
# b = np.random.randint(20, size=(400, 50))
# hit = getHit(b, a)
# mrr = getMRR(b, a)
# print(hit)
# print(mrr)