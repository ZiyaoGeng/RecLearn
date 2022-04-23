"""
Created on Nov 10, 2020
Updated on Apr 23, 2022
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
    def __init__(self, user_num, item_num, embed_dim, mode='inner', w=0.5, use_l2norm=False,
                 loss_name="hinge_loss", gamma=0.5, embed_reg=0., seed=None):
        """AttRec, Sequential Recommendation Model.
        Args:
            :param user_num: An integer type. The largest user index + 1.
            :param item_num: An integer type. The largest item index + 1.
            :param embed_dim: An integer type. Embedding dimension of user vector and item vector.
            :param mode: A string. inner or dist.
            :param w: A float type. The weight of short interest.
            :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
            :param loss_name: A string. You can specify the current point-loss function 'binary_cross_entropy_loss' or
            pair-loss function as 'bpr_loss'、'hinge_loss'.
            :param gamma: A float type. If hinge_loss is selected as the loss function, you can specify the margin.
            :param embed_reg: A float type. The regularizer of embedding.
            :param seed: A Python integer to use as random seed.
        """
        super(AttRec, self).__init__()
        # user embedding
        self.user_embedding = Embedding(input_dim=user_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding, not share embedding
        self.item2_embedding = Embedding(input_dim=item_num,
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
        # norm
        self.use_l2norm = use_l2norm
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_embed = self.user_embedding(tf.reshape(inputs['user'], [-1, ]))  # (None, embed_dim)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), axis=-1)  # (None, seq_len, 1)
        # seq info
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, embed_dim)
        seq_embed *= mask
        # short-term interest
        short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])  # (None, dim)
        # item
        pos_embed = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
        neg_embed = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # item2 embed
        pos_embed2 = self.item2_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
        neg_embed2 = self.item2_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # mode
        if self.mode == 'inner':
            if self.use_l2norm:
                user_embed = tf.math.l2_normalize(user_embed, axis=-1)
                pos_embed = tf.math.l2_normalize(pos_embed, axis=-1)
                neg_embed = tf.math.l2_normalize(neg_embed, axis=-1)
                pos_embed2 = tf.math.l2_normalize(pos_embed2, axis=-1)
                neg_embed2 = tf.math.l2_normalize(neg_embed2, axis=-1)
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
            'click_seq': Input(shape=(100,), dtype=tf.int32),  # suppose sequence length=1
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()
