"""
Created on May 23, 2020

define layer:
    dense layer: DNN Layer
    dice: Data Adaptive Activation Function


@author: Ziyao Geng
"""
import tensorflow as tf


class Dice(tf.keras.layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x
