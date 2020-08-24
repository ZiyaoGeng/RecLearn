"""
Created on August 2, 2020

model: Neural Factorization Machines for Sparse Predictive Analytics

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Concatenate, Dense, Input, BatchNormalization


class NFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., activation='relu', embed_reg=1e-4):
        """
        NFM architecture
        :param feature_columns: a list containing dense and sparse column feature information
        :param hidden_units: a list of dnn hidden units
        :param dnn_dropout: dropout of dnn
        :param activation: activation function of dnn
        :param embed_reg: the regularizer of embedding
        """
        super(NFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dropout = Dropout(rate=dnn_dropout)
        self.bn = BatchNormalization()
        self.concat = Concatenate(axis=-1)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
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
        # BatchNormalization
        x = self.bn(x)
        # Hidden Layers
        for dnn in self.dnn_network:
            x = dnn(x)
        # Dropout
        x = self.dropout(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()