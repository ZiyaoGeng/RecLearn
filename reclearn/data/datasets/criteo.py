"""
Created on July 13, 2020
Updated on Nov 14, 2021
dataset：criteo
features：
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features.
The values of these features have been hashed onto 32 bits for anonymization purposes.
@author: Ziyao Geng(zggzy1996@163.com)
"""
import pickle
import pandas as pd
import os

from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split

from reclearn.data.utils import my_KBinsDiscretizer, splitByLineCount, mkSubFile
from reclearn.data.feature_column import sparseFeature

NAMES = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']


def get_split_file_path(parent_path=None, dataset_path=None, sample_num=5000000):
    """
    get the list of split file path.
    Either parent_path or dataset_path must be valid.
    If exists dataset_path + "/split", parent_path = dataset_path + "/split".
    :param parent_path: A string. split file's parent path.
    :param dataset_path: A string.
    :param sample_num: A int. The sample number of every split file.
    :return: A list. [file1_path, file2_path, ...]
    """
    sub_dir_name = 'split'
    if parent_path is None and dataset_path is None:
        raise ValueError('Please give parent path or file path.')
    if parent_path is None and os.path.exists(os.path.join(os.path.dirname(dataset_path), sub_dir_name)):
        parent_path = os.path.join(os.path.dirname(dataset_path), sub_dir_name)
    elif parent_path is None or not os.path.exists(parent_path):
        splitByLineCount(dataset_path, sample_num, sub_dir_name)
        parent_path = os.path.join(dataset_path, sub_dir_name)
    split_file_name = os.listdir(parent_path)
    split_file_name.sort()
    split_file_list = [parent_path + "/" + file_name for file_name in split_file_name if file_name[-3:] == 'txt']
    return split_file_list


def get_fea_map(fea_map_path=None, split_file_list=None):
    """
    get feature map
    Either parent_path or dataset_path must be valid.
    If exists dir(split_file_list[0]) + "/fea_map.pkl", fea_map_path is valid.
    :param fea_map_path: A string.
    :param split_file_list: A list. [file1_path, file2_path, ...]
    :return: A dict. {'C1':{}, 'C2':{}, ...}
    """
    if fea_map_path is None and split_file_list is None:
        raise ValueError('Please give feature map path or split file list.')
    if fea_map_path is None and os.path.join(os.path.dirname(split_file_list[0]), "fea_map.pkl"):
        fea_map_path = os.path.join(os.path.dirname(split_file_list[0]), "fea_map.pkl")
    if os.path.exists(fea_map_path) and fea_map_path[-3:] == 'pkl':
        with open(fea_map_path, 'rb') as f:
            fea_map = pickle.load(f)
        return fea_map
    fea_map = {}
    for file in tqdm(split_file_list):
        f = open(file)
        for line in f:
            row = line.strip('\n').split('\t')
            for i in range(14, 40):
                if row[i] == '':
                    continue
                name = NAMES[i]
                fea_map.setdefault(name, {})
                if fea_map[name].get(row[i]) is None:
                    fea_map[name][row[i]] = len(fea_map[name])
            for j in range(1, 14):
                if row[j] == '':
                    continue
                name = NAMES[j]
                fea_map.setdefault(name, {})
                fea_map[name].setdefault('min', float(row[j]))
                fea_map[name].setdefault('max', float(row[j]))
                fea_map[name]['min'] = min(fea_map[name]['min'], float(row[j]))
                fea_map[name]['max'] = max(fea_map[name]['max'], float(row[j]))
        f.close()
    for i in range(14, 40):
        fea_map[NAMES[i]]['-1'] = len(fea_map[NAMES[i]])
    fea_map_path = os.path.join(os.path.dirname(split_file_list[0]), "fea_map.pkl")
    with open(fea_map_path, 'wb') as f:
        pickle.dump(fea_map, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return fea_map


def create_criteo_dataset(file, fea_map, embed_dim=8):
    """
    load one split file data.
    :param file: A string. dataset's path
    :param fea_map: A dict.  {'C1':{}, 'C2':{}, ...}
    :param embed_dim: the embedding dimension of sparse features
    :return: feature columns, data
    """
    data_df = pd.read_csv(file, sep='\t', header=None, names=NAMES)
    # data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
    #                       names=NAMES)
    # data_df = data_df.get_chunk(10000)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)
    # map
    for col in sparse_features:
        data_df[col] = data_df[col].map(lambda x: fea_map[col][x])
    # Bin continuous data into intervals.
    data_df[dense_features] = my_KBinsDiscretizer(data_df[dense_features], 1000, fea_map)

    feature_columns = [sparseFeature(feat, len(fea_map[feat]) + 1, embed_dim=embed_dim)
                        for feat in features]

    data_X = {feature: data_df[feature].values.astype('int32') for feature in features}
    data_y = data_df['label'].values.astype('int32')

    return feature_columns, (data_X, data_y)


def create_small_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    load small criteo data without splitting "train.txt".
    :param file: A string. dataset's path.
    :param embed_dim: A scalar. the embedding dimension of sparse features
    :param read_part: A boolean. whether to read part of it
    :param sample_num: A scalar. the number of instances if read_part is True
    :param test_size: A scalar(float). ratio of test dataset
    :return: feature columns, train, test
    """
    if read_part:
        data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                          names=NAMES)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(file, sep='\t', header=None, names=NAMES)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    est = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)

    train_X = {feature: train[feature].values.astype('int32') for feature in features}
    train_y = train['label'].values.astype('int32')
    test_X = {feature: test[feature].values.astype('int32') for feature in features}
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)