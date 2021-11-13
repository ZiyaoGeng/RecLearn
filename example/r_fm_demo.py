"""
Created on August 25, 2020
train FM model
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from reclearn.models.ranking import FM
from reclearn.datasets.criteo import create_criteo_dataset, get_fea_cols
from reclearn.datasets.base import splitByLineCount

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    model = None
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = 'data/criteo/train.txt'
    read_part = False
    sample_num = 5000000
    test_size = 0.2
    k = 8
    learning_rate = 0.001
    batch_size = 4096
    # ========================== Create dataset =======================
    sub_dir_name = "split_file"
    file_path = os.path.join(os.path.dirname(file), sub_dir_name)
    if not os.path.exists(file_path):
        splitByLineCount(file, sample_num, sub_dir_name)
    split_file_list = [file_path + "/" + file_name for file_name in os.listdir(file_path) if file_name[-3:] == 'txt']
    print(split_file_list)
    fea_col_file = os.path.join(file_path, "fea_cols.pkl")
    if not os.path.exists(fea_col_file):
        fea_cols_dict = get_fea_cols(split_file_list)
    else:
        with open(fea_col_file, 'rb') as f:
            fea_cols_dict = pickle.load(f)
    for file in split_file_list:
        print("load %s" % file)
        fea_cols, train, test = create_criteo_dataset(file, fea_cols_dict)
        train_X, train_y = train
        test_X, test_y = test
        # ============================Build Model==========================
        if model is None:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = FM(fea_cols=fea_cols, k=k)
                model.summary()
                # ============================Compile============================
                model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                              metrics=[AUC()])
        # ==============================Fit==============================
        model.fit(
            train_X,
            train_y,
            epochs=1,
            # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
            batch_size=batch_size,
            validation_split=0.1
        )
        # ===========================Test==============================
        print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])