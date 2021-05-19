"""
Created on May 18, 2021

modules of Deep&Crossing: Residual units

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, Layer


class Residual_Units(Layer):
    """
    Residual Units
    """
    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: A list. Neural network hidden units.
        :param dim_stack: A scalar. The dimension of inputs unit.
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs