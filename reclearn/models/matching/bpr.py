"""
Created on Nov 13, 2020
Updated on Nov 06, 2021
Reference: "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI, 2009
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2

from reclearn.models.losses import bpr_loss


class BPR(Model):
    def __init__(self, fea_cols, embed_reg=1e-6, seed=None):
        """
        BPR
        :param fea_cols:  A dict containing {'user_num':, 'item_num:, ...}
        :param embed_reg: A scalar. The regularizer of embedding.
        :param seed: A int scalar.
        """
        super(BPR, self).__init__()
        # user embedding
        self.user_embedding = Embedding(input_dim=fea_cols['user_num'],
                                        input_length=1,
                                        output_dim=fea_cols['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=fea_cols['item_num'],
                                        input_length=1,
                                        output_dim=fea_cols['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_embed = self.user_embedding(inputs['user'])  # (None, embed_dim)
        # item info
        pos_embed = self.item_embedding(inputs['pos_item'])  # (None, embed_dim)
        neg_embed = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # calculate positive item scores and negative item scores
        pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=-1, keepdims=True)  # (None, 1)
        pos_scores = tf.tile(pos_scores, multiples=[1, neg_embed.shape[1]])  # (None, neg_num)
        neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_embed, axis=1), neg_embed), axis=-1)  # (None, neg_num)
        # add loss
        self.add_loss(bpr_loss(pos_scores, neg_scores))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def get_user_vector(self, inputs):
        if len(inputs) < 2 and inputs.get('user') is not None:
            return self.user_embedding(inputs['user'])

    def summary(self):
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()