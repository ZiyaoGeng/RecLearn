"""
Created on July 13, 2020
Updated on Nov 14, 2021
Reference: "Deep & Cross Network for Ad Click Predictions", ADKDD, 2017
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import CrossNetwork, MLP


class DCN(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=0., cross_w_reg=0., cross_b_reg=0.):
        """Deep&Cross Network.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param hidden_units: A list. Neural network hidden units.
            :param activation: A string. Activation function of MLP.
            :param dnn_dropout: A scalar. Dropout of MLP.
            :param embed_reg: A scalar. The regularization coefficient of embedding.
            :param cross_w_reg: A scalar. The regularization coefficient of cross network.
            :param cross_b_reg: A scalar. The regularization coefficient of cross network.
        :return:
        """
        super(DCN, self).__init__()
        self.feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = MLP(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def call(self, inputs):
        # embedding,  (batch_size, embed_dim * fields)
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        x = sparse_embed
        # Cross Network
        cross_x = self.cross_network(x)
        # DNN
        dnn_x = self.dnn_network(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()