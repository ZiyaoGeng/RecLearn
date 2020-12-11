"""
Created on Nov 30, 2020

train MM model

@author: Ziyao Geng
"""
import os
import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

from model import DNN
from evaluate import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    maxlen = 100

    embed_dim = 64
    hidden_unit = 256
    embed_reg = 1e-6  # 1e-6
    activation = 'relu'
    K = 10

    learning_rate = 0.001
    epochs = 30
    batch_size = 512
    # ========================== Create dataset =======================
    item_fea_col, train, val, test = create_implicit_ml_1m_dataset(file, trans_score, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val

    # ============================Build Model==========================
    model = DNN(item_fea_col, maxlen, hidden_unit, activation, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate))

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
            hit_rate, ndcg = evaluate_model(model, test, K)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))
            results.append([epoch + 1, t2 - t1, time() - t2, hit_rate, ndcg])
    # ============================Write============================
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg']).\
        to_csv('log/DNN_log_maxlen_{}_dim_{}_hidden_unit_{}.csv'.
               format(maxlen, embed_dim, hidden_unit), index=False)