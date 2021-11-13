"""
Created on August 26, 2020
Updated on Nov 13, 2021
Reference: "Field-aware Factorization Machines for CTR Prediction", RecSys, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.regularizers import l2

from reclearn.layers import FFM_Layer


class FFM(Model):
    def __init__(self, fea_cols, k, w_reg=1e-6, v_reg=1e-6):
        """
        FFM architecture
        :param fea_cols: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param field_reg_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.fea_cols = fea_cols
        self.ffm = FFM_Layer(self.sparse_feature_columns, k, w_reg, v_reg)

    def call(self, inputs):
        ffm_out = self.ffm(inputs)
        outputs = tf.nn.sigmoid(ffm_out)
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
