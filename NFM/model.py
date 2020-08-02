"""
Created on August 2, 2020

model: Neural Factorization Machines for Sparse Predictive Analytics

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, regularizers
from tensorflow.keras.layers import Embedding, Dropout, Concatenate, Dense, Input, BatchNormalization

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, dropout_rate):
        super(NFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
                                         output_dim=feat['embed_dim'], embeddings_initializer='random_uniform')
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dropout = Dropout(rate=dropout_rate)
        self.bn = BatchNormalization()
        self.concat = Concatenate(axis=-1)
        self.dnn_network = [Dense(units=unit, activation='relu') for unit in hidden_units]
        self.dense = Dense(1)

    def call(self, inputs):
        # Inputs layer
        dense_inputs, sparse_inputs = inputs
        # Embedding layer
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        # Bi-Interaction Layer
        embed = 0.5 * (tf.pow(tf.reduce_sum(embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(embed, 2), axis=1))
        # Concat
        x = self.concat([dense_inputs, embed])
        # Dropout
        x = self.dropout(x)
        # BatchNormalization
        x = self.bn(x)
        # Hidden Layers
        for dnn in self.dnn_network:
            x = dnn(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


def main():
    dense = [{'name': 'c1'}, {'name': 'c2'}, {'name': 'c3'}, {'name': 'c4'}]
    sparse = [{'feat_num': 100, 'embed_dim': 256}, {'feat_num': 200, 'embed_dim': 256}]
    columns = [dense, sparse]
    model = NFM(columns, [200, 200, 200], 0.5)
    model.summary()


main()
