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
    def __init__(self, feature_columns, k=8, w_reg=0., v_reg=0.):
        """Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return
        """
        super(FM, self).__init__()
        self.feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()