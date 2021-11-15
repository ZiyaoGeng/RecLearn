"""
Created on Nov 14, 2021
using small criteo dataset to train the model.
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from reclearn.models.ranking import FM
from reclearn.data.datasets.criteo import create_small_criteo_dataset


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# If you have GPU, and the value is GPU serial number.
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


if __name__ == '__main__':
    # TODO: Hyper Parameters
    file = 'data/criteo/train.txt'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    model_params = {
        'k': 8,
        'w_reg': 0.,
        'v_reg': 0.
    }

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # TODO: Create Dataset
    feature_columns, train, test = create_small_criteo_dataset(file=file,
                                           read_part=read_part,
                                           sample_num=sample_num,
                                           test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # TODO: Build Model
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FM(feature_columns=feature_columns, **model_params)
        model.summary()
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # TODO: Fit
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        batch_size=batch_size,
        validation_split=0.1
    )
    # TODO: Test
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])