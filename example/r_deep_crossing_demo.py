"""
Created on Nov 14, 2021
train Deep Crossing demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from reclearn.models.ranking import Deep_Crossing
from reclearn.data.datasets.criteo import get_split_file_path, get_fea_map, create_criteo_dataset

import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# If you have GPU, and the value is GPU serial number.
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


if __name__ == '__main__':
    # TODO: Hyper Parameters
    file = 'data/criteo/train.txt'
    learning_rate = 0.001
    batch_size = 4096
    embed_dim = 8
    model_params = {
        'hidden_units': [256, 128, 64],
        'dnn_dropout': 0.5,
        'embed_reg': 0.
    }
    # TODO: Split dataset
    # If you want to split the file
    sample_num = 4600000
    split_file_list = get_split_file_path(dataset_path=file, sample_num=sample_num)
    # Or if you split the file before
    # split_file_list = get_split_file_path(parent_path='data/criteo/split')
    print('split file name: %s' % str(split_file_list))
    # TODO: Get Feature Map
    # If you want to make feature map.
    fea_map = get_fea_map(split_file_list=split_file_list)
    # Or if you want to load feature map.
    # fea_map = get_fea_map(fea_map_path='data/criteo/split/fea_map.pkl')
    # TODO: Load test data
    print("load test file: %s" % split_file_list[-1])
    feature_columns, test_data = create_criteo_dataset(split_file_list[-1], fea_map, embed_dim=embed_dim)
    # TODO: Build Model
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Deep_Crossing(feature_columns=feature_columns, **model_params)
        model.summary()
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # TODO: Load train data
    for file in split_file_list[:-1]:
        print("load %s" % file)
        _, train_data = create_criteo_dataset(file, fea_map)
        # TODO: Fit
        model.fit(
            x=train_data[0],
            y=train_data[1],
            epochs=1,
            batch_size=batch_size,
            validation_split=0.1
        )
        # TODO: Test
        train_data = []
        print('test AUC: %f' % model.evaluate(x=test_data[0], y=test_data[1], batch_size=batch_size)[1])