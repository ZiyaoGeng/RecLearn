"""
Created on Nov 10, 2020

modules of AttRec: self-attention mechanism

@author: Ziyao Geng
"""

import tensorflow as tf
import numpy as np
import math

from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.losses import Loss


class SelfAttention_Layer(Layer):
    def __init__(self):
        super(SelfAttention_Layer, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(shape=[self.dim, self.dim], name='weight', 
            initializer='random_uniform')

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        # pos encoding
        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        # Nonlinear transformation
        q = tf.nn.relu(tf.matmul(q, self.W))  # (None, seq_len, dim)
        k = tf.nn.relu(tf.matmul(k, self.W))  # (None, seq_len, dim)
        mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        mask = tf.tile(tf.expand_dims(mask, 1), [1, q.shape[1], 1])  # (None, seq_len, seq_len)
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs, axis=-1)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


