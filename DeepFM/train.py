"""
Created on July 31, 2020

train DeepFM model

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_dataset
from model import DeepFM

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(embed_dim, learning_rate, epochs, batch_size, k, hidden_units, dropout_deep=0.5):
    """
    feature_columns is a list and contains two dict：
    - dense_features: {feat: dense_feature_name}
    - sparse_features: {feat: sparse_feature_name, feat_num: the number of this feature}
    train_X: [dense_train_X, sparse_train_X]
    test_X: [dense_test_X, sparse_test_X]
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset(embed_dim)

    # ============================Build Model==========================
    model = DeepFM(feature_columns, k, hidden_units, dropout_deep=dropout_deep)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
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
    embed_dim = 16
    learning_rate = 0.001
    batch_size = 512
    epochs = 5
    k = 10
    dropout_deep = 0.5
    # The number of hidden units in the deep network layer
    hidden_units = [256, 128, 64]
    main(embed_dim, learning_rate, epochs, batch_size, k, hidden_units, dropout_deep)