"""
Created on Nov 19, 2021
train NCF demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
from time import time
from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import NCF
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg
from reclearn.data.feature_column import sparseFeature

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Hyper parameters
neg_num = 4
embed_dim = 64
learning_rate = 0.001
epochs = 20
batch_size = 512

model_params = {
    'hidden_units': [256, 128, 64],
    'activation': 'relu',
    'dnn_dropout': 0.5,
    'is_batch_norm': False,
    'loss_name': 'binary_entropy_loss',
    'gamma': 0.3
}

k = 10


def main():
    file_path = 'data/ml-1m/ratings.dat'
    # TODO: 1. Split Data
    train_path, val_path, test_path, meta_path = ml.split_movielens(file_path=file_path)
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    # TODO: 2. Build Feature Columns
    fea_cols = {
        'user': sparseFeature('user', max_user_num + 1, embed_dim),
        'item': sparseFeature('item', max_item_num + 1, embed_dim)
    }
    # TODO: 3. Load Data
    train_data = ml.load_ml(train_path, neg_num, max_item_num)
    val_data = ml.load_ml(val_path, neg_num, max_item_num)
    test_data = ml.load_ml(test_path, 100, max_item_num)
    # TODO: 4. Build Model
    model = NCF(fea_cols, **model_params)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    # TODO: 5. Fit Model
    for epoch in range(1, epochs + 1):
        t1 = time()
        model.fit(
            x=train_data,
            epochs=1,
            validation_data=val_data,
            batch_size=batch_size
        )
        if epoch % 2 == 0:
            t2 = time()
            eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], k, batch_size)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f, '
                  % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))


main()