import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_ITEM_NUM = 3953
MAX_USER_NUM = 6041


# general recommendation
def split_data(file_path):
    """split movielens for general recommendation
        Args:
            :param file_path: A string. The file path of 'ratings.dat'.
        :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "ml_train.txt")
    val_path = os.path.join(dst_path, "ml_val.txt")
    test_path = os.path.join(dst_path, "ml_test.txt")
    meta_path = os.path.join(dst_path, "ml_meta.txt")
    users, items = set(), set()
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            user, item, score, timestamp = line.strip().split("::")
            users.add(int(user))
            items.add(int(item))
            history.setdefault(int(user), [])
            history[int(user)].append([item, timestamp])
    random.shuffle(list(users))
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist = history[int(user)]
            hist.sort(key=lambda x: x[1])
            for idx, value in enumerate(hist):
                if idx == len(hist) - 1:
                    f3.write(str(user) + '\t' + value[0] + '\n')
                elif idx == len(hist) - 2:
                    f2.write(str(user) + '\t' + value[0] + '\n')
                else:
                    f1.write(str(user) + '\t' + value[0] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return train_path, val_path, test_path, meta_path


# sequence recommendation
def split_seq_data(file_path):
    """split movielens for sequence recommendation
    Args:
        :param file_path: A string. The file path of 'ratings.dat'.
    :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "ml_seq_train.txt")
    val_path = os.path.join(dst_path, "ml_seq_val.txt")
    test_path = os.path.join(dst_path, "ml_seq_test.txt")
    meta_path = os.path.join(dst_path, "ml_seq_meta.txt")
    users, items = set(), set()
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split("::")
            users.add(int(user))
            items.add(int(item))
            history.setdefault(int(user), [])
            history[int(user)].append([item, timestamp])
        random.shuffle(list(users))
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user, hist in history.items():
            hist_u = history[int(user)]
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(str(user) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + '\n')
            f2.write(str(user) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + "\t" + hist[-2] + '\n')
            f3.write(str(user) + "\t" + ' '.join(hist[:-1]) + "\t" + ' '.join(time[:-1]) + "\t" + hist[-1] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return train_path, val_path, test_path, meta_path


def load_data(file_path, neg_num, max_item_num):
    """load movielens dataset.
    Args:
        :param file_path: A string. The file path.
        :param neg_num: A scalar(int). The negative num of one sample.
        :param max_item_num: A scalar(int). The max index of item.
    :return: A dict. data.
    """
    data = np.array(pd.read_csv(file_path, delimiter='\t'))
    np.random.shuffle(data)
    neg_items = []
    for i in range(len(data)):
        neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
        neg_items.append(neg_item)
    return {'user': data[:, 0].astype(int), 'pos_item': data[:, 1].astype(int), 'neg_item': np.array(neg_items)}


def load_seq_data(file_path, mode, seq_len, neg_num, max_item_num, contain_user=False, contain_time=False):
    """load sequence movielens dataset.
    Args:
        :param file_path: A string. The file path.
        :param mode: A string. "train", "val" or "test".
        :param seq_len: A scalar(int). The length of sequence.
        :param neg_num: A scalar(int). The negative num of one sample.
        :param max_item_num: A scalar(int). The max index of item.
        :param contain_user: A boolean. Whether including user'id input or not.
        :param contain_time: A boolean. Whether including time sequence input or not.
    :return: A dict. data.
    """
    users, click_seqs, time_seqs, pos_items, neg_items = [], [], [], [], []
    with open(file_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if mode == "train":
                user, click_seq, time_seq = line.split('\t')
                click_seq = click_seq.split(' ')
                click_seq = [int(x) for x in click_seq]
                time_seq = time_seq.split(' ')
                time_seq = [int(x) for x in time_seq]
                for i in range(len(click_seq)-1):
                    if i + 1 >= seq_len:
                        tmp = click_seq[i+1-seq_len:i+1]
                        tmp2 = time_seq[i + 1 - seq_len:i + 1]
                    else:
                        tmp = [0] * (seq_len-i-1) + click_seq[:i+1]
                        tmp2 = [0] * (seq_len - i - 1) + time_seq[:i + 1]
                    # gen_neg = _gen_negative_samples(neg_num, click_seq, max_item_num)
                    # neg_item = [neg_item for neg_item in gen_neg]
                    neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                    users.append(int(user))
                    click_seqs.append(tmp)
                    time_seqs.append(tmp2)
                    pos_items.append(click_seq[i + 1])
                    neg_items.append(neg_item)
            else:  # "val", "test"
                user, click_seq, time_seq, pos_item = line.split('\t')
                click_seq = click_seq.split(' ')
                click_seq = [int(x) for x in click_seq]
                time_seq = time_seq.split(' ')
                time_seq = [int(x) for x in time_seq]
                if len(click_seq) >= seq_len:
                    tmp = click_seq[len(click_seq) - seq_len:]
                    tmp2 = time_seq[len(time_seq) - seq_len:]
                else:
                    tmp = [0] * (seq_len - len(click_seq)) + click_seq[:]
                    tmp2 = [0] * (seq_len - len(time_seq)) + time_seq[:]
                # gen_neg = _gen_negative_samples(neg_num, click_seq, max_item_num)
                # neg_item = [neg_item for neg_item in gen_neg]
                neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                users.append(int(user))
                click_seqs.append(tmp)
                time_seqs.append(tmp2)
                pos_items.append(int(pos_item))
                neg_items.append(neg_item)
    data = list(zip(users, click_seqs, time_seqs, pos_items, neg_items))
    random.shuffle(data)
    users, click_seqs, time_seqs, pos_items, neg_items = zip(*data)
    data = {'click_seq': np.array(click_seqs), 'pos_item': np.array(pos_items), 'neg_item': np.array(neg_items)}
    if contain_user:
        data['user'] = np.array(users)
    if contain_time:
        data['time_seq'] = np.array(click_seqs)
    return data


def _gen_negative_samples(neg_num, item_list, max_num):
    for i in range(neg_num):
        # neg = item_list[0]
        # while neg in set(item_list):
        neg = random.randint(1, max_num)
        yield neg


"""
def generate_movielens(file_path, neg_num):
    with open(file_path, 'r') as f:
        for line in f:
            user, pos_item = line.split('\t')
            neg_item = [random.randint(1, MAX_ITEM_NUM) for _ in range(neg_num)]
            yield int(user), int(pos_item), neg_item
"""

"""
def generate_ml(file_path, neg_num):
    return tf.data.Dataset.from_generator(
        generator=generate_movielens,
        args=[file_path, neg_num],
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(neg_num,), dtype=tf.int32),
        )
    )
"""


def generate_seq_data(file_path, seq_len, neg_num):
    with open(file_path, 'r') as f:
        user, hist = f.readline().split('\t')
        hist = hist.split(',')
        hist = [int(item) for item in hist]
        hist_len = len(hist)
        if hist_len < seq_len:
            hist = [0] * (seq_len - hist_len) + hist
            gen_neg = _gen_negative_samples(neg_num, hist, MAX_ITEM_NUM)
            neg_hist = [neg_item for neg_item in gen_neg]
            mask_hist = [0] * (seq_len - hist_len) + [1] * hist_len
        else:
            hist = hist[hist_len - seq_len:]
            gen_neg = _gen_negative_samples(neg_num, hist, MAX_ITEM_NUM)
            neg_hist = [neg_item for neg_item in gen_neg]
            mask_hist = [1] * seq_len
        yield {'user': int(user), 'hist': hist, 'neg_list': neg_hist, 'mask_hist': mask_hist}


def create_ml_1m_dataset(file, trans_score=2, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()
        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num - 1)]
        for i in range(1, len(pos_list)):
            if i == len(pos_list) - 1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = {'user': np.array(train_data['user_id']),
             'pos_item': np.array(train_data['pos_id']),
             'neg_item': np.array(train_data['neg_id']).reshape((-1, 1))}
    val = {'user': np.array(val_data['user_id']),
           'pos_item': np.array(val_data['pos_id']),
           'neg_item': np.array(val_data['neg_id']).reshape((-1, 1))}
    test = {'user': np.array(test_data['user_id']),
            'pos_item': np.array(test_data['pos_id']),
            'neg_item': np.array(test_data['neg_id'])}
    print('============Data Preprocess End=============')
    return train, val, test


def create_seq_ml_1m_dataset(file, trans_score=1, seq_len=40, test_neg_num=100):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param seq_len: A scalar. maxlen.
    :param test_neg_num: A scalar. The number of test negative samples
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start=============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')
    data_df = data_df[data_df.item_count >= 5]
    # trans score
    data_df = data_df[data_df.label >= trans_score]
    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])
    # split dataset and negative sampling
    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data['hist'].append(hist_i)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])
            elif i == len(pos_list) - 2:
                val_data['hist'].append(hist_i)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append([neg_list[i]])
            else:
                train_data['hist'].append(hist_i)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append([neg_list[i]])
    # item feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    # padding
    print('==================Padding===================')
    train = {'click_seq': pad_sequences(train_data['hist'], maxlen=seq_len),
             'pos_item': np.array(train_data['pos_id']),
             'neg_item': np.array(train_data['neg_id'])}
    val = {'click_seq': pad_sequences(val_data['hist'], maxlen=seq_len),
           'pos_item': np.array(val_data['pos_id']),
           'neg_item': np.array(val_data['neg_id'])}
    test = {'click_seq': pad_sequences(test_data['hist'], maxlen=seq_len),
            'pos_item': np.array(test_data['pos_id']),
             'neg_item': np.array(test_data['neg_id'])}
    print('============Data Preprocess End=============')
    return user_num, item_num, train, val, test