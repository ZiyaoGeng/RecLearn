"""
Created on Sept 11, 2020

evaluate model

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np


def getHit(df):
    """
    calculate hit rate
    :return:
    """
    df = df.sort_values('pred_y', ascending=False).reset_index()
    if df[df.true_y == 1].index.tolist()[0] < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    """
    calculate NDCG
    :return:
    """
    df = df.sort_values('pred_y', ascending=False).reset_index()
    i = df[df.true_y == 1].index.tolist()[0]
    if i < _K:
        return np.log(2) / np.log(i+2)
    else:
        return 0.


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    global _K
    _K = K
    test_X, test_y = test
    pred_y = model.predict(test_X)
    test_df = pd.DataFrame(test_y, columns=['true_y'])
    test_df['user_id'] = test_X[0]
    test_df['pred_y'] = pred_y
    tg = test_df.groupby('user_id')
    hit_rate = tg.apply(getHit).mean()
    ndcg = tg.apply(getNDCG).mean()
    return hit_rate, ndcg