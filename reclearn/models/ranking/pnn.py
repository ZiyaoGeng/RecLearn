"""
Created on July 20, 2020
Updated on Nov 13, 2021
Reference: "Product-based Neural Networks for User Response Prediction", ICDM, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Layer, Dropout, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import MLP


class PNN(Model):
    def __init__(self, feature_columns, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=0., w_z_reg=0., w_p_reg=0., l_b_reg=0.):
        """Product-based Neural Networks.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param hidden_units: A list. Neural network hidden units.
            :param mode: A string. 'in' IPNN or 'out' OPNN.
            :param activation: A string. Activation function of MLP.
            :param dnn_dropout: A scalar. Dropout of MLP.
            :param embed_reg: A scalar. The regularization coefficient of embedding.
            :param w_z_reg: A scalar. The regularization coefficient of w_z_ in product layer.
            :param w_p_reg: A scalar. The regularization coefficient of w_p in product layer.
            :param l_b_reg: A scalar. The regularization coefficient of l_b in product layer.
        :return:
        """
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.feature_columns = feature_columns
        # the number of feature fields
        self.field_num = len(self.feature_columns)
        self.embed_dim = self.feature_columns[0]['embed_dim']
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        # parameters
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_normal',
                                   regularizer=l2(w_z_reg),
                                   trainable=True
                                   )
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              hidden_units[0]),
                                       initializer='random_normal',
                                       reguarizer=l2(w_p_reg),
                                       trainable=True)
        # out
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              self.embed_dim, hidden_units[0]),
                                       initializer='random_normal',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        self.l_b = self.add_weight(name='l_b', shape=(hidden_units[0], ),
                                   initializer='random_normal',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        # dnn
        self.dnn_network = MLP(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        # embedding
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        sparse_embed = tf.reshape(sparse_embed, [-1, self.field_num, self.embed_dim])  # (None, filed_num, embed_dim)
        # product layer
        row = []
        col = []
        for i in range(self.field_num - 1):
            for j in range(i + 1, self.field_num):
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
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()