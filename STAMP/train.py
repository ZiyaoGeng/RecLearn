'''
Descripttion: train STAMP model
Author: Ziyao Geng
Date: 2020-10-25 09:27:23
LastEditors: ZiyaoGeng
LastEditTime: 2020-10-26 12:47:08
'''
from time import time
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import os

from model import STAMP
from modules import *
from evaluate import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = '../dataset/Diginetica/train-item-views.csv'
    maxlen = 8
    embed_dim = 100

    K = 20

    learning_rate = 0.005
    batch_size = 128
    epochs = 30
    # ========================== Create dataset =======================
    feature_columns, behavior_list, item_pooling, train, val, test = create_diginetica_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    # ============================Build Model==========================
    model = STAMP(feature_columns, behavior_list, item_pooling, maxlen)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/sas_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    # CrossEntropy()
    # tf.losses.SparseCategoricalCrossentropy()
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=learning_rate))

    for epoch in range(epochs):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y),
            epochs=1,
            # callbacks=[tensorboard, checkpoint],
            batch_size=batch_size,
            )
        # ===========================Test==============================
        t2 = time()
        hit_rate, mrr = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, '
              % (epoch, t2 - t1, time() - t2, hit_rate, mrr))