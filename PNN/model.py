"""
Created on July 20, 2020

model: Product-based Neural Networks for User Response Prediction

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Concatenate, Dense, Layer

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PNN(keras.Model):
    def __init__(self, feature_columns, embed_dim, hidden_units, mode='in'):
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # embedding dimension
        self.embed_dim = embed_dim
        # the number of feature fields
        self.field_num = len(self.sparse_feature_columns)
        # embedding layers
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
                                         output_dim=self.embed_dim, embeddings_initializer='random_uniform')
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # parameters
        self.w_z = self.add_weight(shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform', name='w_z')
        if mode == 'in':
            self.w_p = self.add_weight(shape=(self.field_num, self.field_num,
                                              hidden_units[0]), initializer='random_uniform', name='w_p')
        else:
            self.w_p = self.add_weight(shape=(self.embed_dim, self.embed_dim,
                                              hidden_units[0]), initializer='random_uniform', name='w_p')
        self.l_b = self.add_weight(shape=(hidden_units[0], ), initializer='random_uniform', name='l_b')
        self.concat = Concatenate(axis=-1)
        # dnn
        self.dnn_network = [Dense(units=unit, activation='relu') for unit in hidden_units[1:-1]]
        self.dense_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        z = embed
        if self.mode == 'in':
            p = tf.matmul(embed, tf.transpose(embed, [0, 2, 1]))
        else:
            f_sum = tf.reduce_sum(embed, axis=1, keepdims=True)
            p = tf.matmul(tf.transpose(f_sum, [0, 2, 1]), f_sum)

        l_z = tf.tensordot(z, self.w_z, axes=2)
        l_p = tf.tensordot(p, self.w_p, axes=2)
        l_1 = tf.nn.relu(self.concat([l_z + l_p + self.l_b, dense_inputs]))
        dnn_x = l_1
        for dense in self.dnn_network:
            dnn_x = dense(dnn_x)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()
