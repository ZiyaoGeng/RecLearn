"""
Created on August 20, 2020
Updated on Nov 14, 2021
Reference: "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems", KDD, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, Flatten, Dense, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import Linear, MLP, CIN
from reclearn.layers.utils import index_mapping


class xDeepFM(Model):
    def __init__(self, feature_columns, hidden_units, cin_size, activation='relu', dnn_dropout=0,
                 embed_reg=0., cin_reg=0., w_reg=0.):
        """
        xDeepFM
        :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
        :param hidden_units: A list. Neural network hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param activation: A string. activation function of MLP.
        :param dnn_dropout: A scalar. dropout of MLP.
        :param embed_reg: A scalar. The regularization coefficient of embedding.
        :param cin_reg: A scalar. The regularization coefficient of CIN.
        :param w_reg: A scalar. The regularization coefficient of Linear.
        """
        super(xDeepFM, self).__init__()
        self.feature_columns = feature_columns
        self.embed_dim = self.feature_columns[0]['embed_dim']
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
        self.field_num = len(self.feature_columns)
        self.linear = Linear(self.feature_length, w_reg)
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.mlp = MLP(hidden_units=hidden_units, activation=activation, dnn_dropout=dnn_dropout)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(1, ), initializer=tf.zeros_initializer())

    def call(self, inputs):
        # Linear
        linear_inputs = index_mapping(inputs, self.map_dict)
        linear_inputs = tf.concat([value for _, value in linear_inputs.items()], axis=-1)
        linear_out = self.linear(linear_inputs)  # (batch_size, 1)
        # cin
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        embed_matrix = tf.reshape(sparse_embed, [-1, self.field_num, self.embed_dim])  # (None, filed_num, embed_dim)
        cin_out = self.cin(embed_matrix)  # (batch_size, dim)
        cin_out = self.cin_dense(cin_out)  # (batch_size, 1)
        # dnn
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.mlp(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)  # (batch_size, 1))
        # output
        output = tf.nn.sigmoid(linear_out + cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()