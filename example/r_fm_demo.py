"""
Created on Aug 25, 2020
Updated on Mar 11, 2022
train FM demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from reclearn.models.ranking import FM
from reclearn.data.datasets.criteo import get_split_file_path, get_fea_map, \
    create_criteo_dataset, create_small_criteo_dataset

import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# If you have GPU, and the value is GPU serial number.
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


# TODO: Hyper Parameters
learning_rate = 0.001
batch_size = 4096
model_params = {
    'k': 8,
    'w_reg': 0.,
    'v_reg': 0.
}


def easy_demo(file, sample_num=500000, read_part=True, test_size=0.1, epochs=10):
    feature_columns, train, test = create_small_criteo_dataset(file=file,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # TODO: Build Model
    model = FM(feature_columns=feature_columns, **model_params)
    model.summary()
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # TODO: Fit
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    # TODO: Test
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])


def main(file):
    # TODO: Split dataset
    # If you want to split the file
    sample_num = 4600000
    split_file_list = get_split_file_path(dataset_path=file, sample_num=sample_num)
    # Or if you have split the file before
    # split_file_list = get_split_file_path(parent_path='data/criteo/split')
    print('split file name: %s' % str(split_file_list))
    # TODO: Get Feature Map
    # If you want to make feature map.
    fea_map = get_fea_map(split_file_list=split_file_list)
    # Or if you want to load feature map.
    # fea_map = get_fea_map(fea_map_path='data/criteo/split/fea_map.pkl')
    # TODO: Load test data
    print("load test file: %s" % split_file_list[-1])
    feature_columns, test_data = create_criteo_dataset(split_file_list[-1], fea_map)
    # TODO: Build Model
    model = FM(feature_columns=feature_columns, **model_params)
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
        print('test AUC: %f' % model.evaluate(x=test_data[0], y=test_data[1], batch_size=batch_size)[1])


if __name__ == '__main__':
    file = 'data/criteo/train.txt'
    # easy_demo method only loads sample_num data of the dataset.
    easy_demo(file)
    # main method can train all data.
    # main(file)