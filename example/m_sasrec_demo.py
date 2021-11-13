import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import SASRec
from reclearn.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from time import time


def main():
    epochs = 30
    learning_rate = 0.001
    batch_size = 512
    neg_num = 2
    seq_len = 200
    k = 10
    test_neg_num = 100
    # zip_path = 'data/ml-1m.zip'
    # user_num, item_num, train_path, val_path, test_path = ml.load_zip_seq_movielens(zip_path=zip_path)
    file_path = 'data/ml-1m/ratings.dat'
    user_num, item_num, train_path, val_path, test_path = ml.load_seq_movielens(file_path=file_path)
    train_data = ml.load_seq_ml(train_path, "train", neg_num, seq_len)
    val_data = ml.load_seq_ml(val_path, "val", neg_num, seq_len)
    test_data = ml.load_seq_ml(test_path, "test", test_neg_num, seq_len)
    # user_num, item_num, train_data, val_data, test_data = ml.create_ml_1m_dataset('./data/ml-1m/ratings.dat', trans_score=0, seq_len=seq_len, test_neg_num=100)
    fea_cols = {
        'item_num': item_num,
        'seq_len': seq_len,
    }
    params = {
        'fea_cols': fea_cols,
        'embed_dim': 64,
        'blocks': 2,
        'num_heads': 2,
        'ffn_hidden_unit': 64,
        'dnn_dropout': 0.2,
        'layer_norm_eps': 1e-6,
        'loss_name': 'binary_entropy_loss',
        'embed_reg': 0.
    }
    model = SASRec(**params)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    for epoch in range(1, epochs+1):
        t1 = time()
        model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=1,
            batch_size=batch_size
        )
        eval_dict = eval_pos_neg(model, test_data, batch_size, ["hr", "ndcg"], k)
        t2 = time()
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, '
              % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['ndcg']))


main()