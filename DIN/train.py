"""
Created on May 26, 2020

train model

@author: Ziyao Geng
"""

import tensorflow as tf
import numpy as np
import pickle
import datetime
import os

from model import DIN


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def input_data(dataset, max_sl):
    user = np.array(dataset[:, 0], dtype='int32')
    item = np.array(dataset[:, 1], dtype='int32')
    hist = dataset[:, 2]
    hist_matrix = tf.keras.preprocessing.sequence.pad_sequences(hist, maxlen=max_sl, padding='post')

    sl = np.array(dataset[:, 3], dtype='int32')
    y = np.array(dataset[:, 4], dtype='float32')

    return user, item, hist_matrix, sl, y


def main():
    hidden_unit = 64
    batch_size = 32
    learning_rate = 1
    epochs = 50
    with open('dataset/dataset.pkl', 'rb') as f:
        train_set = np.array(pickle.load(f))
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, max_sl = pickle.load(f)
    train_user, train_item, train_hist, train_sl, train_y = input_data(train_set, max_sl)
    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq=500
    )
    # model checkpoint
    check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                        verbose=1, period=1)

    model = DIN(user_count, item_count, cate_count, cate_list, hidden_unit)
    model.summary()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.1)
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=optimizer, metrics=[tf.keras.metrics.AUC()])
    model.fit(
        [train_user, train_item, train_hist, train_sl],
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[tensorboard, checkpoint]
    )


if __name__ == '__main__':
    main()