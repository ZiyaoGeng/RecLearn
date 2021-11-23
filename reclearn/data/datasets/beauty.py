"""
Created on Nov 23, 2021
Amazon Beauty Dataset.
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

import os
import random
import time
import pandas as pd
import numpy as np

from tqdm import tqdm


# general recommendation
def split_data(file_path):
    """split amazon beauty for general recommendation
        Args:
            :param file_path: A string. The file path of 'ratings.dat'.
        :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "beauty_train.txt")
    val_path = os.path.join(dst_path, "beauty_val.txt")
    test_path = os.path.join(dst_path, "beauty_test.txt")
    meta_path = os.path.join(dst_path, "beauty_meta.txt")
    users, items = set(), dict()
    user_idx, item_idx = 1, 1
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            user, item, score, timestamp = line.strip().split(",")
            users.add(user)
            if items.get(item) is None:
                items[item] = str(item_idx)
                item_idx += 1
            history.setdefault(user, [])
            history[user].append([items[item], timestamp])
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist = history[user]
            if len(hist) < 4:
                continue
            hist.sort(key=lambda x: x[1])
            for idx, value in enumerate(hist):
                if idx == len(hist) - 1:
                    f3.write(str(user_idx) + '\t' + value[0] + '\n')
                elif idx == len(hist) - 2:
                    f2.write(str(user_idx) + '\t' + value[0] + '\n')
                else:
                    f1.write(str(user_idx) + '\t' + value[0] + '\n')
            user_idx += 1
    with open(meta_path, 'w') as f:
        f.write(str(user_idx - 1) + '\t' + str(item_idx - 1))
    return train_path, val_path, test_path, meta_path


# sequence recommendation
def split_seq_data(file_path):
    """split amazon beauty for sequence recommendation
        Args:
            :param file_path: A string. The file path of 'ratings_Beauty.dat'.
        :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "beauty_seq_train.txt")
    val_path = os.path.join(dst_path, "beauty_seq_val.txt")
    test_path = os.path.join(dst_path, "beauty_seq_test.txt")
    meta_path = os.path.join(dst_path, "beauty_seq_meta.txt")
    users, items = set(), dict()
    user_idx, item_idx = 1, 1
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split(",")
            users.add(user)
            if items.get(item) is None:
                items[item] = str(item_idx)
                item_idx += 1
            history.setdefault(user, [])
            history[user].append([items[item], timestamp])
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist_u = history[user]
            if len(hist_u) < 4:
                continue
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(str(user_idx) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + '\n')
            f2.write(str(user_idx) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + "\t" + hist[-2] + '\n')
            f3.write(str(user_idx) + "\t" + ' '.join(hist[:-1]) + "\t" + ' '.join(time[:-1]) + "\t" + hist[-1] + '\n')
            user_idx += 1
    with open(meta_path, 'w') as f:
        f.write(str(user_idx - 1) + '\t' + str(item_idx - 1))
    return train_path, val_path, test_path, meta_path


def load_data(file_path, neg_num, max_item_num):
    """load amazon beauty dataset.
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
    """load sequence be dataset.
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
                        tmp = click_seq[i + 1 - seq_len:i + 1]
                        tmp2 = time_seq[i + 1 - seq_len:i + 1]
                    else:
                        tmp = [0] * (seq_len-i-1) + click_seq[:i + 1]
                        tmp2 = [0] * (seq_len - i - 1) + time_seq[:i + 1]

                    # gen_neg = _gen_negative_samples(neg_num, click_seq, max_item_num)
                    # neg_item = [neg_item for neg_item in gen_neg]
                    neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
                    users.append(int(user))
                    click_seqs.append(tmp)
                    time_seqs.append(tmp2)
                    pos_items.append(click_seq[i + 1])
                    neg_items.append(neg_item)
            else:
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
                users.append(int(user))
                # gen_neg = _gen_negative_samples(neg_num, click_seq, max_item_num)
                # neg_item = [neg_item for neg_item in gen_neg]
                neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
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
