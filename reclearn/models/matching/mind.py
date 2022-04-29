"""
Created on Apr 25, 2022
Reference: "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall", CIKM, 2019
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

from reclearn.layers.core import CapsuleNetwork


class MIND(Model):
    def __init__(self, item_num, embed_dim, seq_len=100, num_interest=4, stop_grad=True, label_attention=True,
                 neg_num=4, batch_size=512, embed_reg=0., seed=None):
        """MIND
        Args:
            :param item_num: An integer type. The largest item index + 1.
            :param embed_dim: An integer type. Embedding dimension of item vector.
            :param seq_len: An integer type. The length of the input sequence.
            :param bilinear_type: An integer type. The number of user interests.
            :param num_interest: An integer type. The number of user interests.
            :param stop_grad: A boolean type. The weights in the capsule network are updated without gradient descent.
            :param label_attention: A boolean type. Whether using label-aware attention or not.
            :param neg_num: A integer type. The number of negative samples for each positive sample.
            :param batch_size: A integer type. The number of samples per batch.
            :param embed_reg: A float type. The regularizer of embedding.
            :param seed: A Python integer to use as random seed.
        :return
        """
        super(MIND, self).__init__()
        with tf.name_scope("Embedding_layer"):
            # item embedding
            self.item_embedding_table = self.add_weight(name='item_embedding_table',
                                                        shape=(item_num, embed_dim),
                                                        initializer='random_normal',
                                                        regularizer=l2(embed_reg),
                                                        trainable=True)
            # embedding bias
            self.embedding_bias = self.add_weight(name='embedding_bias',
                                                  shape=(item_num,),
                                                  initializer=tf.zeros_initializer(),
                                                  trainable=False)
        self.capsule_network = CapsuleNetwork(embed_dim, seq_len, 0, num_interest, stop_grad)
        self.seq_len = seq_len
        self.num_interest = num_interest
        self.label_attention = label_attention
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.neg_num = neg_num
        self.batch_size = batch_size
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs, training=False):
        user_hist_emb = tf.nn.embedding_lookup(self.item_embedding_table, inputs['click_seq'])
        mask = tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32)  # (None, seq_len)
        user_hist_emb = tf.multiply(user_hist_emb, tf.expand_dims(mask, axis=-1))  # (None, seq_len, embed_dim)
        # capsule network
        interest_capsule = self.capsule_network(user_hist_emb, mask)  # (None, num_inter, embed_dim)

        if training:
            if self.label_attention:
                item_embed = tf.nn.embedding_lookup(self.item_embedding_table, tf.reshape(inputs['pos_item'], [-1, ]))
                inter_att = tf.matmul(interest_capsule, tf.reshape(item_embed, [-1, self.embed_dim, 1]))  # (None, num_inter, 1)
                inter_att = tf.nn.softmax(tf.pow(tf.reshape(inter_att, [-1, self.num_interest]), 1))

                user_info = tf.matmul(tf.reshape(inter_att, [-1, 1, self.num_interest]), interest_capsule)  # (None, 1, embed_dim)
                user_info = tf.reshape(user_info, [-1, self.embed_dim])
            else:
                user_info = tf.reduce_max(interest_capsule, axis=1)  # (None, embed_dim)
            # train, sample softmax loss
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=self.item_embedding_table,
                biases=self.embedding_bias,
                labels=tf.reshape(inputs['pos_item'], shape=[-1, 1]),
                inputs=user_info,
                num_sampled=self.neg_num * self.batch_size,
                num_classes=self.item_num
            ))
            # add loss
            self.add_loss(loss)
            return loss
        else:
            # predict/eval
            pos_info = tf.nn.embedding_lookup(self.item_embedding_table, inputs['pos_item'])  # (None, embed_dim)
            neg_info = tf.nn.embedding_lookup(self.item_embedding_table, inputs['neg_item'])  # (None, neg_num, embed_dim)

            if self.label_attention:
                user_info = tf.reduce_max(interest_capsule, axis=1)  # (None, embed_dim)
            else:
                user_info = tf.reduce_max(interest_capsule, axis=1)  # (None, embed_dim)

            # calculate similar scores.
            pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info),
                                       axis=-1)  # (None, neg_num)
            logits = tf.concat([pos_scores, neg_scores], axis=-1)
            return logits

    def summary(self):
        inputs = {
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()

