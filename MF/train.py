"""
Created on August 31, 2020

train model

@author: Ziyao Geng
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import MF
from utils import create_explicit_ml_1m_dataset

import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = '../dataset/ml-1m/ratings.dat'
    test_size = 0.2

    latent_dim = 32
    # use bias
    use_bias = True

    learning_rate = 0.001
    batch_size = 512
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = create_explicit_ml_1m_dataset(file, latent_dim, test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = MF(feature_columns, use_bias)
    model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/mf_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),
                    metrics=['mse'])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test rmse: %f' % np.sqrt(model.evaluate(test_X, test_y)[1]))
