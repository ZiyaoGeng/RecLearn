"""
Created on Oct 23, 2020

model: STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation

@author: Ziyao Geng
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, \
    Dropout, Embedding, Flatten, Input

from modules import *


class STAMP(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, item_pooling, activation='tanh',
                 maxlen=40, embed_reg=1e-4):
        """
        STAMP
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param item_pooling: A Ndarray or Tensor, shape=(m, n),
        m is the number of items, and n is the number of behavior feature. The item pooling.
        :param activation: A String. The activation of FFN.
        :param maxlen: A scalar. Number of length of sequence.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(STAMP, self).__init__()
        self.maxlen = maxlen
        # item pooling
        self.item_pooling = item_pooling
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.seq_len = len(behavior_feature_list)

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]
        # Attention
        self.attention_layer = Attention_Layer(d=self.sparse_feature_columns[0]['embed_dim'])
        # FNN, hidden unit must be equal to embedding dimension
        self.ffn1 = Dense(self.sparse_feature_columns[0]['embed_dim'], activation=activation)
        self.ffn2 = Dense(self.sparse_feature_columns[0]['embed_dim'], activation=activation)

    def call(self, inputs):
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs

        seq_inputs = tf.concat([seq_inputs, tf.expand_dims(item_inputs, axis=-1)], axis=-1)
        x = dense_inputs
        # other
        for i in range(self.other_sparse_len):
            x = tf.concat([x, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)
        # seq
        seq_embed, m_t, item_pooling_embed = None, None, None
        for i in range(self.seq_len):
            seq_embed = self.embed_seq_layers[i](seq_inputs[:, i]) if seq_embed is None \
                else seq_embed + self.embed_seq_layers[i](seq_inputs[:, i])
            m_t = self.embed_seq_layers[i](item_inputs[:, i]) if m_t is None \
                else m_t + self.embed_seq_layers[i](item_inputs[:, i])  # (None, d)
            item_pooling_embed = self.embed_seq_layers[i](self.item_pooling[:, i]) \
                if item_pooling_embed is None \
                else item_pooling_embed + self.embed_seq_layers[i](self.item_pooling[:, i])
        m_s = tf.reduce_sum(seq_embed, axis=1)  # (None, d)
        # attention
        m_a = self.attention_layer([seq_embed, m_s, m_t])  # (None, d)
        # try to add other embedding vector
        if self.other_sparse_len != 0 or self.dense_len != 0:
            m_a = tf.concat([m_a, x], axis=-1)
            m_t = tf.concat([m_t, x], axis=-1)
        # FFN
        h_s = self.ffn1(m_a)  # (None, d)
        h_t = self.ffn2(m_t)  # (None, d)
        # Calculate
        z = tf.matmul(tf.multiply(tf.expand_dims(h_t, axis=1), item_pooling_embed), tf.expand_dims(h_s, axis=-1))
        z = tf.squeeze(z, axis=-1)  # (None, m)
        # Outputs
        outputs = tf.nn.softmax(z)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len,), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.seq_len, self.maxlen), dtype=tf.int32)
        item_inputs = Input(shape=(self.seq_len,), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()


def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    item_pooling = tf.constant([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    features = [dense_features, sparse_features]
    model = STAMP(features, behavior_list, item_pooling)
    model.summary()


test_model()
