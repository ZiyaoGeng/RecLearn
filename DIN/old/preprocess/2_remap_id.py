"""
Created on May 24, 2020

reviews_df保留'reviewerID'【用户ID】, 'asin'【产品ID】, 'unixReviewTime'【浏览时间】三列
meta_df保留'asin'【产品ID】, 'categories'【种类】两列

reviews.pkl: 1689188 * 9
['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText',
       'overall', 'summary', 'unixReviewTime', 'reviewTime']

       reviewerID        asin  ... unixReviewTime   reviewTime
0   AO94DHGC771SJ  0528881469  ...     1370131200   06 2, 2013
1   AMO214LNFCEI4  0528881469  ...     1290643200  11 25, 2010

meta.pkl: 63001 * 9
['asin', 'imUrl', 'description', 'categories', 'title', 'price',
       'salesRank', 'related', 'brand']
         asin                                         categories
0  0528881469  [[Electronics, GPS & Navigation, Vehicle GPS, ...

@author: Ziyao Geng
"""

import random
import pickle
import numpy as np
import pandas as pd

random.seed(2020)


def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


# reviews
reviews_df = pd.read_pickle('../raw_data/reviews.pkl')
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# meta
meta_df = pd.read_pickle('../raw_data/meta.pkl')
meta_df = meta_df[['asin', 'categories']]
# 类别只保留最后一个
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# meta_df文件的物品ID映射
asin_map, asin_key = build_map(meta_df, 'asin')
# meta_df文件物品种类映射
cate_map, cate_key = build_map(meta_df, 'categories')
# reviews_df文件的用户ID映射
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

# user_count: 192403	item_count: 63001	cate_count: 801	example_count: 1689188
user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
#       (user_count, item_count, cate_count, example_count))

# 按物品id排序，并重置索引
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

# reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# 各个物品对应的类别
cate_list = np.array(meta_df['categories'], dtype='int32')

# 保存所需数据为pkl文件
with open('../raw_data/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
