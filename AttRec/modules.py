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
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    def positional_encoding(self, QK_input):
        encoded_vec = [pos / np.power(10000.0, 2 * i / self.dim)
                       for pos in range(QK_input.shape[1]) for i in range(self.dim)]
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        encoded_vec = tf.reshape(tf.convert_to_tensor(encoded_vec, dtype=tf.float32), shape=(-1, self.dim))

        return encoded_vec