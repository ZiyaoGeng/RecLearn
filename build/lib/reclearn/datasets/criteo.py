"""
Created on July 13, 2020
dataset：criteo dataset sample
features：
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features.
The values of these features have been hashed onto 32 bits for anonymization purposes.
@author: Ziyao Geng(zggzy1996@163.com)
"""
import pickle
import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

from reclearn.datasets.base import sparseFeature, my_KBinsDiscretizer


def get_fea_cols(file_list):
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
    fea_dict = {}
    for file in tqdm(file_list):
        f = open(file)
        for line in f:
            row = line.strip('\n').split('\t')
            for i in range(14, 40):
                if row[i] == '':
                    continue
                name = names[i]
                fea_dict.setdefault(name, {})
                if fea_dict[name].get(row[i]) is None:
                    fea_dict[name][row[i]] = len(fea_dict[name])
            for j in range(1, 14):
                if row[j] == '':
                    continue
                name = names[j]
                fea_dict.setdefault(name, {})
                fea_dict[name].setdefault('min', float(row[j]))
                fea_dict[name].setdefault('max', float(row[j]))
                fea_dict[name]['min'] = min(fea_dict[name]['min'], float(row[j]))
                fea_dict[name]['max'] = max(fea_dict[name]['max'], float(row[j]))
        f.close()
    for i in range(14, 40):
        fea_dict[names[i]]['-1'] = len(fea_dict[names[i]])
    fea_col_path = os.path.join(os.path.dirname(file_list[0]), "fea_cols.pkl")
    with open(fea_col_path, 'wb') as f:
        pickle.dump(fea_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return fea_dict


def create_criteo_dataset(file, fea_cols_dict, embed_dim=8, test_size=0.2):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param fea_cols_dict:
    :param embed_dim: the embedding dimension of sparse features
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
    data_df = pd.read_csv(file, sep='\t', header=None, names=names)
    # data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
    #                       names=names)
    # data_df = data_df.get_chunk(100)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)
    # map
    for col in sparse_features:
        data_df[col] = data_df[col].map(lambda x: fea_cols_dict[col][x])
    # Bin continuous data into intervals.
    data_df[dense_features] = my_KBinsDiscretizer(data_df[dense_features], 100, fea_cols_dict)

    feature_columns = [sparseFeature(feat, len(fea_cols_dict[feat]) + 1, embed_dim=embed_dim)
                        for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)
    #
    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')
    # data_X = train[features].values.astype('int32')
    # data_y = train['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)
    # return feature_columns, (data_X, data_y)



def generate_criteo(file_path, fea_cols_dict):
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']
    f = open(file, 'r')
    for line in f:
        row = line.strip('\n').split('\t')