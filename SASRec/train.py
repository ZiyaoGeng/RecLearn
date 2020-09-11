"""
Created on Sept 11, 2020

train SASRec model

@author: Ziyao Geng
"""
from time import time
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import os

from model import SASRec
from evaluate import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    maxlen = 200
    embed_dim = 32

    K = 10
    blocks = 2
    num_heads = 1
    ffn_hidden_unit = 256
    dropout = 0.2
    norm_training = True
    causality = False

    learning_rate = 0.001
    batch_size = 512
    epochs = 5
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_implicit_ml_1m_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    # ============================Build Model==========================
    model = SASRec(feature_columns, behavior_list, blocks, num_heads, ffn_hidden_unit, dropout,
                   maxlen, norm_training, causality)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/sas_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate))

    for epoch in range(epochs):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train_X,
            train_y,
            validation_data=[val_X, val_y],
            epochs=1,
            # callbacks=[tensorboard, checkpoint],
            batch_size=batch_size,
            )
        # ===========================Test==============================
        t2 = time()
        hit_rate, ndcg = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, '
              % (epoch, t2 - t1, time() - t2, hit_rate, ndcg))