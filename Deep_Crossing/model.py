"""
Created on July 27, 2020

model: Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dense, ReLU, Layer

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Residual_Units(Layer):
    """
    Residual Units
    """
    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: the dimension of cross layer unit
        :param dim_stack: the dimension of inputs unit
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack)
        self.relu = ReLU()

    def call(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class Deep_Crossing(keras.Model):
    def __init__(self, feature_columns, hidden_units):
        super(Deep_Crossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
                                         output_dim=feat['embed_dim'], embeddings_initializer='random_uniform')
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # the total length of embedding
        embed_dim = sum([feat['embed_dim'] for feat in self.sparse_feature_columns])
        # the dimension of stack layer
        dim_stack = len(self.dense_feature_columns) + embed_dim
        # hidden_units = [512, 512, 256, 128, 64]
        self.res_network = [Residual_Units(unit, dim_stack) for unit in hidden_units]
        self.dense = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        stack = dense_inputs
        for i in range(sparse_inputs.shape[1]):
            embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            stack = tf.concat([stack, embed_i], axis=-1)
        r = stack
        for res in self.res_network:
            r = res(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


def main():
    """
    test model
    :return:
    """
    dense = [{'name': 'c1'}, {'name': 'c2'}, {'name': 'c3'}, {'name': 'c4'}]
    sparse = [{'feat_num': 100, 'embed_dim': 256}, {'feat_num': 200, 'embed_dim': 256}]
    columns = [dense, sparse]
    model = Deep_Crossing(columns, 256, [512, 512, 256, 128, 64])
    model.summary()


# main()