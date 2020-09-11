"""
Created on May 25, 2020

create implicit ml-1m dataset

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np
import random
import pickle
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_implicit_ml_1m_dataset(file, embed_dim=8, maxlen=40):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                     names=['user_id', 'item_id', 'label', 'Timestamp'])
    # implicit dataset
    data_df.loc[data_df.label >= 2, 'label'] = 1
    data_df.loc[data_df.label < 2, 'label'] = 0
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    train_data, val_data, test_data = [], [], []

    item_count = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_count)
                return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + 100)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data.append([hist_i, pos_list[i], [user_id, 1]])
                for neg in neg_list[i:]:
                    test_data.append([hist_i, neg, [user_id, 0]])
                # test_data.append([hist_i, [pos_list[i]] + neg_list[i:], pos_list[i]])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, pos_list[i], 1])
                val_data.append([hist_i, neg_list[i], 0])
            else:
                train_data.append([hist_i, pos_list[i], 1])
                train_data.append([hist_i, neg_list[i], 0])

    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [[],
                       [sparseFeature('item_id', item_num, embed_dim)]]
    behavior_list = ['item_id']
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])
    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               np.expand_dims(pad_sequences(train['hist'], maxlen=maxlen), axis=1),
               train['target_item'].values]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
               np.expand_dims(pad_sequences(val['hist'], maxlen=maxlen), axis=1),
               val['target_item'].values]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
               np.expand_dims(pad_sequences(test['hist'], maxlen=maxlen), axis=1),
               test['target_item'].values]
    test_y = test['label'].values.tolist()
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)


# create_implicit_ml_1m_dataset('../dataset/ml-1m/ratings.dat')