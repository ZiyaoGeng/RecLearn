"""
Created on August 2, 2020
Updated on Nov 14, 2021
Reference: "Neural Factorization Machines for Sparse Predictive Analytics", SIGIR, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, Layer, Dense, Input, BatchNormalization
from tensorflow.keras.regularizers import l2

from reclearn.layers import MLP


class NFM(Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., activation='relu', bn_use=True, embed_reg=0.):
        """Neural Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param hidden_units: A list. Neural network hidden units.
            :param activation: A string. Activation function of dnn.
            :param dnn_dropout: A scalar. Dropout of dnn.
            :param bn_use: A Boolean. Use BatchNormalization or not.
            :param embed_reg: A scalar. The regularization coefficient of embedding.
        :return:
        """
        super(NFM, self).__init__()
        self.feature_columns = feature_columns
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        self.embed_dim = self.feature_columns[0]['embed_dim']
        self.field_num = len(self.feature_columns)
        self.bn = BatchNormalization()
        self.bn_use = bn_use
        self.dnn_network = MLP(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        # Embedding layer, (batch_size, fields * embed_dim)
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        sparse_embed = tf.reshape(sparse_embed, [-1, self.field_num, self.embed_dim])  # (None, filed_num, embed_dim)
        # Bi-Interaction Layer
        sparse_embed = 0.5 * (tf.pow(tf.reduce_sum(sparse_embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(sparse_embed, 2), axis=1))  # (None, embed_dim)
        # Concat
        x = sparse_embed
        # BatchNormalization
        if self.bn_use:
            x = self.bn(x)
        # Hidden Layers
        x = self.dnn_network(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()