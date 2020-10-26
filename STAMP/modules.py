'''
Descripttion: 
Author: Ziyao Geng
Date: 2020-10-23 11:10:08
LastEditors: ZiyaoGeng
LastEditTime: 2020-10-26 09:57:35
'''
"""
Created on Oct 23, 2020

modules of STAMP: attention mechanism

@author: Ziyao Geng
"""
import tensorflow as tf

from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer


class Attention_Layer(Layer):
    """
    Attention Layer
    """
    def __init__(self, d, reg=1e-4):
        """

        :param d: A scalar. The dimension of embedding.
        :param reg: A scalar. The regularizer of parameters
        """
        self.d = d
        self.reg = reg
        super(Attention_Layer, self).__init__()

    def build(self, input_shape):
        self.W0 = self.add_weight(name='W0',
                                  shape=(self.d, 1),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W3 = self.add_weight(name='W3',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.b = self.add_weight(name='b',
                                  shape=(self.d,),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)

    def call(self, inputs):
        seq_embed, m_s, x_t = inputs
        """
        seq_embed: (None, seq_len, d)
        W1: (d, d)
        x_t: (None, d)
        W2: (d, d)
        m_s: (None, d)
        W3: (d, d)
        W0: (d, 1)
        """
        alpha = tf.matmul(tf.nn.sigmoid(
            tf.tensordot(seq_embed, self.W1, axes=[2, 0]) + tf.expand_dims(tf.matmul(x_t, self.W2), axis=1) +
            tf.expand_dims(tf.matmul(m_s, self.W3), axis=1) + self.b), self.W0)
        m_a = tf.reduce_sum(tf.multiply(alpha, seq_embed), axis=1)  # (None, d)
        return m_a


# class CrossEntropy(Loss):
#     def call(self, y_true, y_pred):
#         y_true = tf.one_hot(tf.squeeze(tf.cast(y_true, dtype=tf.int32), axis=-1), depth=y_pred.shape[-1])
#         return - (tf.math.log(tf.reduce_sum(y_pred * y_true)) + tf.reduce_sum(tf.math.log(1.0 - y_pred)) - \
#             tf.math.log(1 - tf.reduce_sum(y_pred * y_true)))