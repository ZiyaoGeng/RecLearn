"""
Created on Sept 11, 2020

train Caser model

@author: Ziyao Geng
"""
import os
import tensorflow as tf
from time import time
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.optimizers import Adam

from model import Caser
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
    maxlen = 200

    embed_dim = 32
    hor_n = 8
    hor_h = 2
    ver_n = 8
    dropout = 0.5
    activation = 'relu'
    embed_reg = 1e-5
    K = 10

    learning_rate = 0.001
    epochs = 30
    batch_size = 512
    # ========================== Create dataset =======================
    feature_columns, train, val, test = create_implicit_ml_1m_dataset(file, trans_score, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val

    # ============================Build Model==========================
    model = Caser(feature_columns, maxlen, hor_n, hor_h, ver_n, dropout, activation, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y),
            epochs=1,
            batch_size=batch_size,
        )
        # ===========================Test==============================
        t2 = time()
        if epoch % 5 == 0:
            hit_rate, ndcg, mrr = evaluate_model(model, test, K)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG= %.4f, , MRR = %.4f'
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr))
            results.append([epoch + 1, t2 - t1, time() - t2, hit_rate, ndcg, mrr])
    # ============================Write============================
    # pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg']).\
    #     to_csv('log/Caser_log_maxlen_{}_dim_{}_hor_n_{}_ver_n_{}_K_{}_.csv'.
    #            format(maxlen, embed_dim, hor_n, ver_n, K), index=False)