"""
Created on July 9, 2020
Updated on Nov 13, 2021
Reference: "Wide & Deep Learning for Recommender Systems", DLRS, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import Linear, MLP
from reclearn.layers.utils import index_mapping


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=0., w_reg=0.):
        """
        Wide&Deep
        :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of MLP.
        :param dnn_dropout: A scalar. Dropout of MLP.
        :param embed_reg: A scalar. The regularization coefficient of embedding.
        :param w_reg: A scalar. The regularization coefficient of Linear.
        """
        super(WideDeep, self).__init__()
        self.feature_columns = feature_columns
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        self.dnn_network = MLP(hidden_units, activation, dnn_dropout)
        self.linear = Linear(self.feature_length, w_reg=w_reg)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs):
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        x = sparse_embed  # (batch_size, field * embed_dim)
        # Wide
        wide_inputs = index_mapping(inputs, self.map_dict)
        wide_inputs = tf.concat([value for _, value in wide_inputs.items()], axis=-1)
        wide_out = self.linear(wide_inputs)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()