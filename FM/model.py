"""
Created on August 25, 2020
Updated on May, 18, 2021

model: Factorization Machines

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2


class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        # mapping
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs


class FM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()