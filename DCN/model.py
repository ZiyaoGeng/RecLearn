"""
Created on July 13, 2020

model: Deep & Cross Network for Ad Click Predictions

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Concatenate, Dense, Layer, Dropout


class CrossNetwork(Layer):
    """
    Cross Network
    """
    def __init__(self, layer_num, reg_w=1e-4):
        """
        :param layer_num: the deep of cross network
        :param reg_w: the regularizer of w
        """
        self.layer_num = layer_num
        self.reg_w = reg_w
        super(CrossNetwork, self).__init__()

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_uniform',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(shape=(dim, 1),
                            initializer='random_uniform',
                            name='b_'+str(i))
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l
        x_l = tf.squeeze(x_l, axis=2)
        return x_l


class DCN(keras.Model):
    """
    Deep&Cross Network model
    """
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.,
                 activation='relu', embed_reg=1e-4, cross_reg=1e-4):
        """
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param hidden_units: a list of neural network hidden units
        :param dnn_dropout: dropout of dnn
        :param activation: activation function of dnn
        :param embed_reg: the regularizer of embedding
        :param cross_reg: the regularizer of cross network
        """
        super(DCN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_reg)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.concat = Concatenate(axis=-1)
        self.dense_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        x = dense_inputs
        for i in range(sparse_inputs.shape[1]):
            embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            x = tf.concat([x, embed_i], axis=-1)
        cross_x = self.cross_network(x)
        dnn_x = x
        for dense in self.dnn_network:
            dnn_x = dense(dnn_x)
        dnn_x = self.dropout(dnn_x)
        x = self.concat([cross_x, dnn_x])
        outputs = tf.nn.sigmoid(self.dense_final(x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()
