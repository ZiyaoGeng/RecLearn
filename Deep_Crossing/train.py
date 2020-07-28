"""
Created on July 27, 2020

train Deep Crossing model

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_dataset
from model import Deep_Crossing

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(learning_rate, epochs, hidden_units):
    """
    feature_columns is a list and contains two dict：
    - dense_features: {feat: dense_feature_name}
    - sparse_features: {feat: sparse_feature_name, feat_num: the number of this feature}
    train_X: [dense_train_X, sparse_train_X]
    test_X: [dense_test_X, sparse_test_X]
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset()

    # ============================Build Model==========================
    model = Deep_Crossing(feature_columns, hidden_units)
    model.summary()
    # ============================model checkpoint======================
    check_path = 'save/deep_crossing_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                    verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[checkpoint],
        batch_size=128,
        validation_split=0.2
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


if __name__ == '__main__':
    epochs = 10
    learning_rate = 0.001
    # The number of hidden units in the deep network layer
    hidden_units = [512, 512, 256, 128, 64]
    main(learning_rate, epochs, hidden_units)