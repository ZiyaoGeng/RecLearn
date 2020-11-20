"""
Created on Nov 10, 2020

model: Next Item Recommendation with Self-Attentive Metric Learning

@author: Ziyao Geng
"""

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2

from modules import *


class AttRec(Model):
    def __init__(self, feature_columns, maxlen=40, mode='inner', gamma=0.5, w=0.5, embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param gamma: A scalar. if mode == 'dist', gamma is the margin.
        :param mode: A string. inner or dist.
        :param w: A scalar. The weight of short interest.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(AttRec, self).__init__()
        # maxlen
        self.maxlen = maxlen
        # w
        self.w = w
        self.gamma = gamma
        self.mode = mode
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # user embedding
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding
        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        # pos embedding
        self.pos_embedding = Embedding(input_dim=self.maxlen,
                                       input_length=1,
                                       output_dim=self.embed_dim,
                                       mask_zero=False,
                                       embeddings_initializer='random_uniform',
                                       embeddings_regularizer=l2(embed_reg))
        # self-attention
        self.self_attention = SelfAttention_Layer()

    def call(self, inputs, **kwargs):
        # input
        user_inputs, seq_inputs, pos_inputs, neg_inputs = inputs
        # mask
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)  # (None, maxlen)
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # item
        pos_embed = self.item_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed = self.item_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)
        # item2 embed
        pos_embed2 = self.item2_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed2 = self.item2_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)
        # pos embedding
        # seq_embed += tf.expand_dims(self.positional_encoding(seq_embed), axis=0)
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding

        # short-term interest
        short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])  # (None, dim)

        # mode
        if self.mode == 'inner':
            # long-term interest, pos and neg
            pos_long_interest = tf.multiply(user_embed, pos_embed)
            neg_long_interest = tf.multiply(user_embed, neg_embed)
            # combine
            pos_scores = self.w * tf.reduce_sum(tf.multiply(short_interest, pos_embed), axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True)  # (None, 1)
            neg_scores = self.w * tf.reduce_sum(tf.multiply(short_interest, neg_embed), axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True)  # (None, 1)
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            # distance
            # long-term interest, pos and neg
            pos_long_interest = tf.square(user_embed - pos_embed2)  # (None, dim)
            neg_long_interest = tf.square(user_embed - neg_embed2)  # (None, dim)
            # combine. Here is a difference from the original paper.
            pos_scores = self.w * tf.reduce_sum(tf.square(short_interest - pos_embed), axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True)  # (None, 1)
            neg_scores = self.w * tf.reduce_sum(tf.square(short_interest - neg_embed), axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True)  # (None, 1)
            # minimize loss
            # self.add_loss(tf.reduce_sum(tf.nn.relu(neg_scores - pos_scores + self.gamma)))
            self.add_loss(tf.reduce_sum(tf.maximum(neg_scores - pos_scores + self.gamma, 0)))
        return pos_scores, neg_scores

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, pos_inputs, neg_inputs], 
            outputs=self.call([user_inputs, seq_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    seq_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, seq_features]
    model = AttRec(features, mode='dist')
    model.summary()


# test_model()