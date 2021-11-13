"""
Created on August 25, 2020
Updated on Nov, 11, 2021
Reference: "Factorization Machines", ICDM, 2010
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import FM_Layer


class FM(Model):
    def __init__(self, fea_cols, k, w_reg=1e-8, v_reg=1e-8):
        """
        Factorization Machines
        :param fea_cols: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.fea_cols = fea_cols
        self.fm = FM_Layer(fea_cols, k, w_reg, v_reg)

    def call(self, inputs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()