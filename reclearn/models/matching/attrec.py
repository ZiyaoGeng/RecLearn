"""
Created on Nov 10, 2020
Updated on Nov 11, 2021
Reference: "Next Item Recommendation with Self-Attentive Metric Learning", AAAI, 2019
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import SelfAttention
from reclearn.models.losses import get_loss


class AttRec(Model):
    def __init__(self, fea_cols, embed_dim=16, mode='inner', loss_name="bpr_loss", gamma=0.5, w=0.5, embed_reg=1e-8, seed=None):
        """
        AttRec
        :param fea_col: A dict contains 'item_num', 'seq_len' and 'embed_dim'.
        :param embed_dim: A scalar. The dimension of embedding for user, item and other features.
        :param gamma: A scalar. if mode == 'dist', gamma is the margin.
        :param mode: A string. inner or dist.
        :param loss_name: A string. You can specify the current pair-loss function as "bpr_loss" or "hinge_loss".
        :param gamma: A scalar. If hinge_loss is selected as the loss function, you can specify the margin.
        :param w: A scalar. The weight of short interest.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param seed: A int scalar.
        """
        super(AttRec, self).__init__()
        # user embedding
        self.user_embedding = Embedding(input_dim=fea_cols['user_num'],
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=fea_cols['item_num'],
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding, not share embedding
        self.item2_embedding = Embedding(input_dim=fea_cols['item_num'],
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # self-attention
        self.self_attention = SelfAttention()
        # w
        self.w = w
        # mode
        self.mode = mode
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seq_len
        self.seq_len = fea_cols['seq_len']
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), axis=-1)  # (None, seq_len, 1)
        # user info
        user_embed = self.user_embedding(inputs['user'])  # (None, embed_dim)
        # seq info
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, embed_dim)
        # short-term interest
        short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])  # (None, dim)
        # item
        pos_embed = self.item_embedding(inputs['pos_item'])  # (None, embed_dim)
        neg_embed = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # item2 embed
        pos_embed2 = self.item2_embedding(inputs['pos_item'])  # (None, embed_dim)
        neg_embed2 = self.item2_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # mode
        if self.mode == 'inner':
            # long-term interest, pos and neg
            pos_long_interest = tf.multiply(user_embed, pos_embed2)  # (None, embed_dim)
            neg_long_interest = tf.multiply(tf.expand_dims(user_embed, axis=1), neg_embed2)  # (None, neg_num, embed_dim)
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(tf.expand_dims(short_interest, axis=1), neg_embed), axis=-1)
        else:
            # clip by norm
            user_embed = tf.clip_by_norm(user_embed, 1, -1)
            pos_embed = tf.clip_by_norm(pos_embed, 1, -1)
            neg_embed = tf.clip_by_norm(neg_embed, 1, -1)
            pos_embed2 = tf.clip_by_norm(pos_embed2, 1, -1)
            neg_embed2 = tf.clip_by_norm(neg_embed2, 1, -1)
            # distance, long-term interest, pos and neg
            pos_long_interest = tf.square(user_embed - pos_embed2)  # (None, embed_dim)
            neg_long_interest = tf.square(tf.expand_dims(user_embed, axis=1) - neg_embed2)  # (None, neg_num, embed_dim)
            # combine. Here is a difference from the original paper.
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(tf.expand_dims(short_interest, axis=1) - neg_embed), axis=-1)
        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()
