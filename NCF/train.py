"""
Created on Dec 20, 2020

train NCF model

@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
import pandas as pd
import tensorflow as tf
from time import time
from tensorflow.keras.optimizers import Adam

from model import NCF
from evaluate import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'

    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    test_neg_num = 100

    embed_dim = 32
    hidden_units = [256, 128, 64]
    embed_reg = 1e-6  # 1e-6
    activation = 'relu'
    dropout = 0.2
    K = 10

    learning_rate = 0.001
    epochs = 20
    batch_size = 512

    # ========================== Create dataset =======================
    feature_columns, train, val, test = create_ml_1m_dataset(file, trans_score, embed_dim, test_neg_num)

    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = NCF(feature_columns, hidden_units, dropout, activation, embed_reg)
        model.summary()
        # =========================Compile============================
        model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train,
            None,
            validation_data=(val, None),
            epochs=1,
            batch_size=batch_size,
        )
        # ===========================Test==============================
        t2 = time()
        if epoch % 2 == 0:
            hit_rate, ndcg = evaluate_model(model, test, K)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
            results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg])
    # ========================== Write Log ===========================
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
        .to_csv('log/NCF_log_dim_{}__K_{}.csv'.format(embed_dim, K), index=False)