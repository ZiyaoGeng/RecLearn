"""
Created on Nov 10, 2020

create implicit ml-1m dataset(update, delete dense_inputs, sparse_inputs)

This dataset is for BPR model use.

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: A string, feature name.
    :param feat_num: A scalar. The total number of sparse features that do not repeat.
    :param embed_dim: A scalar, embedding dimension.
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_implicit_ml_1m_dataset(file, trans_score=2, embed_dim=8):
    """
    Create implicit ml-1m dataset.
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # implicit dataset
    data_df.loc[data_df.label < trans_score, 'label'] = 0
    data_df.loc[data_df.label >= trans_score, 'label'] = 1

    # sort by time user and timestamp
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    # create train, val, test data
    train_data, val_data, test_data = [], [], []

    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
                return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + 100)]
        for i in range(1, len(pos_list)):
            if i == len(pos_list) - 1:
                for neg in neg_list[i:]:
                    test_data.append([[user_id], [pos_list[i]], [neg]])
            elif i == len(pos_list) - 2:
                val_data.append([[user_id], [pos_list[i]], [neg_list[i]]])
            else:
                train_data.append([[user_id], [pos_list[i]], [neg_list[i]]])

    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [[sparseFeature('user_id', user_num, embed_dim)],
                       [sparseFeature('item_id', item_num, embed_dim)]]

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'pos_item', 'neg_item'])
    val = pd.DataFrame(val_data, columns=['user_id', 'pos_item', 'neg_item'])
    test = pd.DataFrame(test_data, columns=['user_id', 'pos_item', 'neg_item'])

    # create dataset
    def df_to_list(data):
        return [np.array(data['user_id'].tolist()),
            np.array(data['pos_item'].tolist()), np.array(data['neg_item'].tolist())]

    train_X = df_to_list(train)
    val_X = df_to_list(val)
    test_X = df_to_list(test)
    print('============Data Preprocess End=============')
    return feature_columns, train_X, val_X, test_X


# create_implicit_ml_1m_dataset('../dataset/ml-1m/ratings.dat')