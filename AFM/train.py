"""
Created on August 3, 2020

train AFM model

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_dataset
from model import AFM

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(sample_num, embed_dim, learning_rate, epochs, batch_size, mode='max', attention_hidden_unit=None):
    """

    :param sample_num: the num of training sample
    :param embed_dim: the dimension of all embedding layer
    :param learning_rate:
    :param epochs:
    :param batch_size:
    :param mode
    :param attention_hidden_unit:
    :return:
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset(sample_num, embed_dim)

    # ============================Build Model==========================
    model = AFM(feature_columns, mode, attention_hidden_unit=attention_hidden_unit)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/afm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
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
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


if __name__ == '__main__':
    sample_num = 100000
    embed_dim = 16
    learning_rate = 0.001
    batch_size = 512
    epochs = 5
    attention_hidden_unit = 64
    # main(sample_num, embed_dim, learning_rate, epochs, batch_size, mode='max')
    # main(sample_num, embed_dim, learning_rate, epochs, batch_size, mode='avg')
    main(sample_num, embed_dim, learning_rate, epochs, batch_size,
         mode='att', attention_hidden_unit=attention_hidden_unit)