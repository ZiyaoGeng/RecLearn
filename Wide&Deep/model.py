"""
Created on July 9, 2020

model: Wide & Deep Learning for Recommender Systems

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Dropout, Input
from tensorflow.keras.regularizers import l2


class Linear(layers.Layer):
    """
    Linear Part
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result


class DNN(layers.Layer):
    """
    Deep part
    """
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        """

        :param hidden_units: list of hidden layer units's numbers
        :param activation: activation function
        :param dnn_dropout: dropout number
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu', dnn_dropout=0., embed_reg=1e-4):
        """
        Wide&Deep architecture
        :param feature_columns: a list containing dense and sparse column feature information
        :param hidden_units: a list of dnn hidden units
        :param activation: activation function of dnn
        :param dnn_dropout: dropout of dnn
        :param embed_reg: the regularizer of embedding
        """
        super(WideDeep, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        stack = dense_inputs
        for i in range(sparse_inputs.shape[1]):
            embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            stack = tf.concat([stack, embed_i], axis=-1)

        # Wide
        wide_out = self.linear(dense_inputs)
        # Deep
        deep_out = self.dnn(stack)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()
