"""
Created on Sept 10, 2020

model: Self-Attentive Sequential Recommendation

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, \
    Dropout, Embedding, Flatten, Input

from modules import *


class SASRec(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=40, norm_training=True, causality=False, embed_reg=1e-4):
        """

        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param blocks: A scalar. The Number of blocks.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param maxlen: A scalar. Number of length of sequence
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        :param embed_reg: A scalar. The regularizer of embedding
        """
        super(SASRec, self).__init__()
        self.maxlen = maxlen
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.feq_len = len(behavior_feature_list)
        self.item_embed = self.sparse_feature_columns[0]['embed_dim']
        self.d_model = self.item_embed
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] in behavior_feature_list]

        self.attention_block = [SelfAttentionBlock(self.d_model, num_heads, ffn_hidden_unit,
                                                   dropout, norm_training, causality) for b in range(blocks)]
        self.dense = Dense(self.item_embed, activation='relu')

    def call(self, inputs):
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        x = dense_inputs
        for i in range(self.other_sparse_len):
            x = tf.concat([x, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)
        pos_encoding = tf.expand_dims(self.positional_encoding(seq_inputs), axis=0)
        seq_embed, item_embed = None, None
        for i in range(self.feq_len):
            seq_embed = self.embed_seq_layers[i](seq_inputs[:, i]) if seq_embed is None \
                else seq_embed + self.embed_seq_layers[i](seq_inputs[:, i])
            item_embed = self.embed_seq_layers[i](item_inputs[:, i]) if item_embed is None \
                else item_embed + self.embed_seq_layers[i](item_inputs[:, i])

        seq_embed += pos_encoding
        att_outputs = seq_embed
        for block in self.attention_block:
            att_outputs = block(att_outputs)

        att_outputs = Flatten()(att_outputs)

        x = tf.concat([att_outputs, x], axis=-1)
        x = self.dense(x)
        # Predict
        outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(x, item_embed), axis=1, keepdims=True))
        return outputs

    def positional_encoding(self, seq_inputs):
        encoded_vec = [pos / tf.pow(10000, 2 * i / tf.cast(self.d_model, dtype=tf.float32))
                       for pos in range(seq_inputs.shape[-1]) for i in range(self.item_embed)]
        encoded_vec[::2] = tf.sin(encoded_vec[::2])
        encoded_vec[1::2] = tf.cos(encoded_vec[1::2])
        encoded_vec = tf.reshape(tf.convert_to_tensor(encoded_vec, dtype=tf.float32), shape=(-1, self.item_embed))

        return encoded_vec

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len, ), dtype=tf.int32)
        seq_inputs = Input(shape=(self.feq_len, self.maxlen), dtype=tf.int32)
        item_inputs = Input(shape=(self.feq_len, ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()


def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = SASRec(features, behavior_list, num_heads=8)
    model.summary()


test_model()
