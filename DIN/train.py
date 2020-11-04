'''
Descripttion: 
Author: Ziyao Geng
Date: 2020-10-26 22:18:12
LastEditors: ZiyaoGeng
LastEditTime: 2020-11-04 23:23:15
'''
"""
Created on Oct 23, 2020

train DIN model

@author: Ziyao Geng
"""
import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from model import DIN
from utils import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = 'raw_data/remap.pkl'
    maxlen = 20
    
    embed_dim = 8
    att_hidden_units= [80, 40] 
    ffn_hidden_units= [256, 128, 64]
    dnn_dropout = 0.5
    att_activation='sigmoid'
    ffn_activation='prelu'

    learning_rate = 0.001
    batch_size = 4096
    epochs = 5
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_amazon_electronic_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test
    # ============================Build Model==========================
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation, 
        ffn_activation, maxlen, dnn_dropout)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(val_X, val_y),
        batch_size=batch_size,
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])