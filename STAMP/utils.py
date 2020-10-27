'''
Descripttion: create Diginetica dataset
Author: Ziyao Geng
Date: 2020-10-23 19:52:53
LastEditors: ZiyaoGeng
LastEditTime: 2020-10-27 10:00:03
'''
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
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


def convert_sequence(data_df):
    """
    :param data_df: train, val or test
    """
    data_sequence = []
    for sessionId, df in tqdm(data_df[['sessionId', 'itemId']].groupby(['sessionId'])):
        item_list = df['itemId'].tolist()

        for i in range(1, len(item_list)):
            hist_i = item_list[:i]
            # hist_item, next_click_item(label)
            data_sequence.append([hist_i, item_list[i]])

    return data_sequence

def create_diginetica_dataset(file, embed_dim=8, maxlen=40):
    """
    :param file: A string. dataset path
    :param embed_dim: A scalar. latent factor
    :param maxlen: A scalar. 
    :return: feature_columns, behavior_list, train, val, test
    """
    print('==========Data Preprocess Start============')
    # load dataset
    data_df = pd.read_csv(file, sep=";") # (1235380, 5)
    
    # filter out sessions of length of 1
    data_df['session_count'] = data_df.groupby('sessionId')['sessionId'].transform('count')
    data_df = data_df[data_df.session_count > 1]  # (1144686, 6)

    # filter out items that appear less than 5 times
    data_df['item_count'] = data_df.groupby('itemId')['itemId'].transform('count')
    data_df = data_df[data_df.item_count >= 5]  # (1004834, 7)

    # label encoder itemId, {0, 1, ..., }
    le = LabelEncoder()
    data_df['itemId'] = le.fit_transform(data_df['itemId'])
    
     # sorted by eventdate, sessionId
    data_df = data_df.sort_values(by=['eventdate', 'sessionId'])

    # split dataset, 1 day for valdation, 7 days for test
    train = data_df[data_df.eventdate < '2016-05-25']  # (916485, 7)
    val = data_df[data_df.eventdate == '2016-05-25']  # (10400, 7)
    test = data_df[data_df.eventdate > '2016-05-25']  # (77949, 7)

    # convert sequence
    train = pd.DataFrame(convert_sequence(train), columns=['hist', 'label'])
    val = pd.DataFrame(convert_sequence(val), columns=['hist', 'label'])
    test = pd.DataFrame(convert_sequence(test), columns=['hist', 'label'])
    
    # Padding
    # not have dense inputs and other sparse inputs
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               np.expand_dims(pad_sequences(train['hist'], maxlen=maxlen), axis=1)]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
               np.expand_dims(pad_sequences(val['hist'], maxlen=maxlen), axis=1)]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
               np.expand_dims(pad_sequences(test['hist'], maxlen=maxlen), axis=1)]
    test_y = test['label'].values

    # item pooling
    item_pooling = np.sort(data_df['itemId'].unique().reshape(-1, 1), axis=0)

    # feature columns, dense feature columns + sparse feature columns
    item_num = data_df['itemId'].max() + 1
    feature_columns = [[],
                       [sparseFeature('item_id', item_num, embed_dim)]]

    # behavior list
    behavior_list = ['item_id']

    print('===========Data Preprocess End=============')
    
    return feature_columns, behavior_list, item_pooling, (train_X, train_y), (val_X, val_y), (test_X, test_y)
    

# create_diginetica_dataset('../dataset/Diginetica/train-item-views.csv')