"""
Created on July 13, 2020

dataset：criteo dataset sample
features：
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features.
The values of these features have been hashed onto 32 bits for anonymization purposes.
@author: Ziyao Geng
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}


def create_dataset():
    data_df = pd.read_csv('dataset/criteo_sample.txt', sep=',')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=4)
                        for feat in sparse_features]]

    train, test = train_test_split(data_df, test_size=0.2)

    train_X = [train[dense_features].values, train[sparse_features].values.astype('int32')]
    train_y = train['label'].values

    test_X = [test[dense_features].values, test[sparse_features].values.astype('int32')]
    test_y = test['label'].values

    return feature_columns, train_X, test_X, train_y, test_y