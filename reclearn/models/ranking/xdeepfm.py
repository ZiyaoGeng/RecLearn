"""
Created on August 20, 2020
Reference: "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems", KDD, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Flatten, Dense, Input

from reclearn.layers import Linear, DNN, CIN


class xDeepFM(Model):
    def __init__(self, fea_cols, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-6, cin_reg=1e-6, w_reg=1e-6):
        """
        xDeepFM
        :param fea_cols: A list. sparse column feature information.
        :param hidden_units: A list. a list of dnn hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param dnn_dropout: A scalar. dropout of dnn.
        :param dnn_activation: A string. activation function of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cin_reg: A scalar. The regularizer of cin.
        :param w_reg: A scalar. The regularizer of Linear.
        """
        super(xDeepFM, self).__init__()
        self.fea_cols = fea_cols
        self.embed_dim = self.fea_cols[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.fea_cols)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.linear = Linear(self.feature_length, w_reg)
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(1, ), initializer=tf.zeros_initializer())

    def call(self, inputs):
        # Linear
        linear_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        linear_out = self.linear(linear_inputs)  # (batch_size, 1)
        # cin
        embed = [self.embed_layers['embed_{}'.format(i)](inputs[:, i]) for i in range(inputs.shape[1])]
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)  # (batch_size, dim)
        cin_out = self.cin_dense(cin_out)  # (batch_size, 1)
        # dnn
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)  # (batch_size, 1))
        # output
        output = tf.nn.sigmoid(linear_out + cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()