"""
Created on May 24, 2020

json ---> pd --->pkl
meta只保留reviews文件中出现过的商品

@author: Ziyao Geng
"""

import pickle
import pandas as pd


def to_df(file_path):
    """
    转化为DataFrame结构
    :param file_path: 文件路径
    :return:
    """
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


reviews_df = to_df('../raw_data/reviews_Electronics_5.json')

# 改变列的顺序
# reviews2_df = pd.read_json('../raw_data/reviews_Electronics_5.json', lines=True)


with open('../raw_data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../raw_data/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('../raw_data/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
