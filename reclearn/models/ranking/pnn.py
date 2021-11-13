"""
Created on July 20, 2020
Updated on Nov 13, 2021
Reference: "Product-based Neural Networks for User Response Prediction", ICDM, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Layer, Dropout, Input

from reclearn.layers import MLP


class PNN(Model):
    def __init__(self, fea_cols, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=1e-6, w_z_reg=1e-6, w_p_reg=1e-6, l_b_reg=1e-6):
        """
        Product-based Neural Networks
        :param fea_cols: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param mode: A string. 'in' IPNN or 'out'OPNN.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param w_z_reg: A scalar. The regularizer of w_z_ in product layer
        :param w_p_reg: A scalar. The regularizer of w_p in product layer
        :param l_b_reg: A scalar. The regularizer of l_b in product layer
        """
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.fea_cols = fea_cols
        # the number of feature fields
        self.field_num = len(self.fea_cols)
        self.embed_dim = self.fea_cols[0]['embed_dim']
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # parameters
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform',
                                   regularizer=l2(w_z_reg),
                                   trainable=True
                                   )
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              hidden_units[0]),
                                       initializer='random_uniform',
                                       reguarizer=l2(w_p_reg),
                                       trainable=True)
        # out
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        self.l_b = self.add_weight(name='l_b', shape=(hidden_units[0], ),
                                   initializer='random_uniform',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        # dnn
        self.dnn_network = MLP(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        sparse_inputs = inputs
        sparse_embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        sparse_embed = tf.transpose(tf.convert_to_tensor(sparse_embed), [1, 0, 2])  # (None, field_num, embed_dim)
        # product layer
        row = []
        col = []
        for i in range(len(self.sparse_feature_columns) - 1):
            for j in range(i + 1, len(self.sparse_feature_columns)):
                row.append(i)
                col.append(j)
        p = tf.gather(sparse_embed, row, axis=1)
        q = tf.gather(sparse_embed, col, axis=1)
        if self.mode == 'in':
            l_p = tf.tensordot(p*q, self.w_p, axes=2)  # (None, hidden[0])
        else:  # out
            u = tf.expand_dims(q, 2)  # (None, field_num(field_num-1)/2, 1, emb_dim)
            v = tf.expand_dims(p, 2)  # (None, field_num(field_num-1)/2, 1, emb_dim)
            l_p = tf.tensordot(tf.matmul(tf.transpose(u, [0, 1, 3, 2]), v), self.w_p, axes=3)  # (None, hidden[0])

        l_z = tf.tensordot(sparse_embed, self.w_z, axes=2)  # (None, hidden[0])
        l_1 = tf.nn.relu(tf.concat([l_z + l_p + self.l_b], axis=-1))
        # dnn layer
        dnn_x = self.dnn_network(l_1)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()