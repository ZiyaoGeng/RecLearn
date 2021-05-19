"""
Created on August 26, 2020
Updated on May 19, 2021

model: Field-aware Factorization Machines for CTR Prediction

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.regularizers import l2

from modules import FFM_Layer


class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        FFM architecture
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param field_reg_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(self.sparse_feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        ffm_out = self.ffm(inputs)
        outputs = tf.nn.sigmoid(ffm_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()


