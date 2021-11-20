"""
Created on Nov 20, 2021
train PopRec demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
from time import time
from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import PopRec
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

k = 10


def main():
    # TODO: 1. Split Data
    # file_path = 'data/ml-1m/ratings.dat'
    # train_path, val_path, test_path, _ = ml.split_movielens(file_path=file_path)
    train_path = 'data/ml-1m/ml_train.txt'
    val_path = 'data/ml-1m/ml_val.txt'
    test_path = 'data/ml-1m/ml_test.txt'
    meta_path = 'data/ml-1m/ml_meta.txt'
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    # TODO: 2. Load Data
    test_data = ml.load_ml(test_path, 100, max_item_num)
    # TODO: 3. Build Model
    model = PopRec(train_path=train_path, delimiter='\t')
    # TODO: 3. Update Model.
    model.update(data_path=val_path, delimiter='\t')
    # TODO: 4. Evaluate Model
    t1 = time()
    eval_dict = eval_pos_neg(model, test_data, metric_names=['hr', 'mrr', 'ndcg'], k=k)
    print('Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f, '
          % (time() - t1, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))


main()