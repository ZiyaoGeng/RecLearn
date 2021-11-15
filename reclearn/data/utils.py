"""
Created on July 13, 2020
Updated on May 18, 2021
input feature columns: sparseFeature, denseFeature
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
import time
import numpy as np


def mkSubFile(lines, head, srcName, sub_dir_name, sub):
    root_path, file = os.path.split(srcName)
    file_name, suffix = file.split('.')
    split_file_name = file_name + "_" + str(sub).zfill(2) + "." + suffix
    split_file = os.path.join(root_path, sub_dir_name, split_file_name)
    if not os.path.exists(os.path.join(root_path, sub_dir_name)):
        os.mkdir(os.path.join(root_path, sub_dir_name))
    print('make file: %s' % split_file)
    f = open(split_file, 'w')
    try:
        f.writelines([head])
        f.writelines(lines)
        return sub + 1
    finally:
        f.close()


def splitByLineCount(filename, count, sub_dir_name):
    f = open(filename, 'r')
    try:
        head = f.readline()
        buf = []
        sub = 1
        for line in f:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf, head, filename, sub_dir_name, sub)
                buf = []
        if len(buf) != 0:
            mkSubFile(buf, head, filename, sub_dir_name, sub)
    finally:
        f.close()


def my_KBinsDiscretizer(data_df, n_bins, min_max_dict, strategy='uniform'):
    features = data_df.columns
    n_features = len(features)
    bin_edges = np.zeros(n_features, dtype=object)
    for idx, feature in enumerate(features):
        if strategy == 'uniform':
            bin_edges[idx] = np.linspace(min_max_dict[feature]['min'], min_max_dict[feature]['max'], n_bins + 1)
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * np.abs(data_df[feature])
        np.digitize(data_df[feature] + eps, bin_edges[idx][1:])
    return data_df