"""
Created on July 27, 2020
Updated on Nov 13, 2021
Reference: "Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features", KDD, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input

from reclearn.layer import Residual_Units


class Deep_Crossing(Model):
    def __init__(self, fea_cols, hidden_units, res_dropout=0., embed_reg=1e-6):
        """
        Deep&Crossing
        :param fea_cols: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param res_dropout: A scalar. Dropout of resnet.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Deep_Crossing, self).__init__()
        self.fea_cols = fea_cols
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # the total length of embedding layers
        embed_layers_len = sum([feat['embed_dim'] for feat in self.sparse_feature_columns])
        self.res_network = [Residual_Units(unit, embed_layers_len) for unit in hidden_units]
        self.res_dropout = Dropout(res_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        r = sparse_embed
        for res in self.res_network:
            r = res(r)
        r = self.res_dropout(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()