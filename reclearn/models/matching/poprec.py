"""
Created on Nov 20, 2021
Model: Pop Recommendation
@author: Ziyao Geng(zggzy1996@163.com)
"""
import numpy as np
import pandas as pd

from tqdm import tqdm

from reclearn.evaluator.metrics import *


class PopRec:
    def __init__(self, train_path, delimiter='\t'):
        """Pop recommendation
        Args:
            :param train_data: A String. The path of data, such as "*.txt" or "*.csv".
            :param delimiter: A character. Please give field delimiter.
        :return:
        """
        self.item_count = dict()
        self.pop_item_list = list()
        self.max_count = 0
        self.__build_item_count_dict(train_path, delimiter)

    def __build_item_count_dict(self, train_path, delimiter='\t'):
        data = np.array(pd.read_csv(train_path, delimiter=delimiter))
        for i in tqdm(range(len(data))):
            user, item = data[i]
            self.item_count.setdefault(int(item), 0)
            self.item_count[int(item)] += 1
            self.max_count = max(self.item_count[int(item)], self.max_count)
        # sorting
        self.pop_item_list = [x[0] for x in sorted(self.item_count.items(), key=lambda x: x[1], reverse=True)]

    def update(self, data_path, delimiter='\t'):
        """Update
        Args:
            :param data_path: A String. The path of data, such as "*.txt" or "*.csv".
            :param delimiter: A character. Please give field delimiter.
        :return:
        """
        self.__build_item_count_dict(data_path, delimiter)

    def clear(self):
        self.item_count = dict()
        self.pop_item_list = list()
        self.__build_item_count_dict(train_path, delimiter)

    def predict(self, test_data, batch_size=None):
        """predict recommended items
        :param test_data: A dict.
        :param batch_size: None.
        :return: A recommendation list of length k.
        """
        pos_item_list, neg_items_list = test_data['pos_item'], test_data['neg_item']
        pos_item_list = np.reshape(pos_item_list, (-1, 1))
        item_list = np.hstack((pos_item_list, neg_items_list))
        pred_item_list = [list(map(lambda x: self.item_count.get(x, -1) / self.max_count, l)) for l in item_list]
        return np.array(pred_item_list)

    def evaluate(self, test_path, k, metric_names, delimiter='\t'):
        """evaluate PopRec
        Args:
            :param test_path: A String. The path of data, such as "*.txt" or "*.csv".
            :param k: A scalar(int).
            :param metric_names: A list like ['hr'].
            :param delimiter: A character. Please give field delimiter.
        :return: A result dict such as {'hr':, 'ndcg':, ...}
        """
        data = np.array(pd.read_csv(test_path, delimiter=delimiter))
        pred_items = self.predict(k)
        rank = []
        for i in range(len(data)):
            user, item = data[i]
            # if item in pred_items
            try:
                rank.append(pred_items.index(item))
            except:
                rank.append(k+1)
        res_dict = {}
        for name in metric_names:
            if name == 'hr':
                res = hr(rank, k)
            elif name == 'ndcg':
                res = ndcg(rank, k)
            elif name == 'mrr':
                res = mrr(rank, k)
            else:
                break
            res_dict[name] = res
        return res_dict