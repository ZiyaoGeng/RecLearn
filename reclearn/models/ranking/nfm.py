"""
Created on August 2, 2020
Reference: "Neural Factorization Machines for Sparse Predictive Analytics", SIGIR, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Layer, Dense, Input, BatchNormalization

from reclearn.layers import MLP


class NFM(Model):
    def __init__(self, fea_cols, hidden_units, dnn_dropout=0., activation='relu', bn_use=True, embed_reg=1e-6):
        """
        NFM architecture
        :param fea_cols: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param bn_use: A Boolean. Use BatchNormalization or not.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NFM, self).__init__()
        self.fea_cols = fea_cols
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.fea_cols)
        }
        self.bn = BatchNormalization()
        self.bn_use = bn_use
        self.dnn_network = MLP(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        # Inputs layer
        sparse_inputs = inputs
        # Embedding layer
        sparse_embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        sparse_embed = tf.transpose(tf.convert_to_tensor(sparse_embed), [1, 0, 2])  # (None, filed_num, embed_dim)
        # Bi-Interaction Layer
        sparse_embed = 0.5 * (tf.pow(tf.reduce_sum(sparse_embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(sparse_embed, 2), axis=1))  # (None, embed_dim)
        # Concat
        x = sparse_embed
        # BatchNormalization
        x = self.bn(x, training=self.bn_use)
        # Hidden Layers
        x = self.dnn_network(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()