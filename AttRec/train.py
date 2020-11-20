"""
Created on Nov 11, 2020

train AttRec model

@author: Ziyao Geng
"""
import os
import pandas as pd
import tensorflow as tf
from time import time
from tensorflow.keras.optimizers import Adam

from model import AttRec
from modules import *
from evaluate import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'

    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    maxlen = 50
    
    embed_dim = 32
    embed_reg = 1e-6  # 1e-6
    gamma = 0.5
    mode = 'inner'  # 'inner' or 'dist'
    w = 0.5
    K = 10

    learning_rate = 0.001
    epochs = 40
    batch_size = 512
    # ========================== Create dataset =======================
    feature_columns, train, val, test = create_implicit_ml_1m_dataset(file, trans_score, embed_dim, maxlen)
    train_X = train
    val_X = val

    # ============================Build Model==========================
    model = AttRec(feature_columns, maxlen, mode, gamma, w, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train_X,
            None,
            validation_data=(val_X, None),
            epochs=1,
            # callbacks=[tensorboard, checkpoint],
            batch_size=batch_size,
            )
        # ===========================Test==============================
        t2 = time()
        if epoch % 5 == 0:
            hit_rate, ndcg, mrr = evaluate_model(model, test, K)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, MRR = %.4f'
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr))
            results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr])
        # ========================== Write Log ===========================
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time',
                                   'hit_rate', 'ndcg', 'mrr']).to_csv(
        'log/AttRec_log_maxlen_{}_dim_{}_K_{}_w_{}.csv'.format(maxlen, embed_dim, K, w), index=False)