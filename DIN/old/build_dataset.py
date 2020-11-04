"""
Created on May 25, 2020

构造训练集、测试集

@author: Ziyao Geng
"""
import random
import pickle

random.seed(2020)

with open('raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set, test_set = [], []

# 最大的序列长度
max_sl = 0

"""
生成训练集、测试集，每个用户所有浏览的物品（共n个）前n-1个为训练集（正样本），并生成相应的负样本，每个用户
共有n-2个训练集（第1个无浏览历史），第n个作为测试集。
故测试集共有192403个，即用户的数量。训练集共2608764个
"""
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    # 每个用户浏览过的物品，即为正样本
    pos_list = hist['asin'].tolist()
    max_sl = max(max_sl, len(pos_list))

    # 生成负样本
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    # 正负样本比例1：1
    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        # 生成每一次的历史记录，即之前的浏览历史
        hist = pos_list[:i]
        sl = len(hist)
        if i != len(pos_list) - 1:
            # 保存正负样本，格式：用户ID，正/负物品id，浏览历史，浏览历史长度，标签（1/0）
            train_set.append((reviewerID, pos_list[i], hist, sl, 1))
            train_set.append((reviewerID, neg_list[i], hist, sl, 0))
        else:
            # 最后一次保存为测试集
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, sl, label))

# 打乱顺序
random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

# 写入dataset.pkl文件
with open('dataset/dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, max_sl), f, pickle.HIGHEST_PROTOCOL)
