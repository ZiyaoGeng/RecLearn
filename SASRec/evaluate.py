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
    test_df = pd.DataFrame(test_y, columns=['user_id', 'true_y'])
    test_df['pred_y'] = pred_y
    user_num = len(test_df['user_id'].unique())
    hit = test_df.groupby('user_id').apply(getHit)
    hit_rate = hit.sum() / user_num
    ndcg = test_df.groupby('user_id').apply(getNDCG)
    ndcg = ndcg.sum() / user_num
    return hit_rate, ndcg