"""
Created on Nov 18, 2020
Updated on Apr 23, 2022
Reference: "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding", WSDM, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2

from reclearn.models.losses import get_loss


class Caser(Model):
    def __init__(self, user_num, item_num, embed_dim, seq_len=100, hor_n=8, hor_h=2, ver_n=4,
                 activation='relu', dnn_dropout=0., use_l2norm=False,
                 loss_name="binary_cross_entropy_loss", gamma=0.5, embed_reg=0, seed=None):
        """Caser, Sequential Recommendation Model.
        Args:
            :param user_num: An integer type. The largest user index + 1.
            :param item_num: An integer type. The largest item index + 1.
            :param embed_dim: An integer type. Embedding dimension of user vector and item vector.
            :param seq_len: An integer type. The length of the input sequence.
            :param hor_n: An integer type. The number of horizontal filters.
            :param hor_h: An integer type. Height of horizontal filters.
            :param ver_n: An integer type. The number of vertical filters.
            :param activation: A string. Activation function name of user and item MLP layer.
            :param dnn_dropout: Float between 0 and 1. Dropout of user and item MLP layer.
            :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
            :param loss_name: A string. You can specify the current point-loss function 'binary_cross_entropy_loss' or
            pair-loss function as 'bpr_loss'、'hinge_loss'.
            :param gamma: A float type. If hinge_loss is selected as the loss function, you can specify the margin.
            :param embed_reg: A float type. The regularizer of embedding.
            :param seed: A Python integer to use as random seed.
        :return:
        """
        super(Caser, self).__init__()
        # user embedding
        self.user_embedding = Embedding(input_dim=user_num,
                                        input_length=1,
                                        output_dim=embed_dim // 2,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim // 2,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding
        self.item2_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # seq_len
        self.seq_len = seq_len
        # horizontal filters
        self.hor_n = hor_n
        self.hor_h = hor_h if hor_h <= self.seq_len else self.seq_len
        # vertical filters
        self.ver_n = ver_n
        self.ver_w = 1
        # horizontal conv
        self.hor_conv_list = [
            Conv1D(filters=i+1, kernel_size=self.hor_h) for i in range(self.hor_n + 1)
        ]
        # vertical conv, should transpose
        self.ver_conv = Conv1D(filters=self.ver_n, kernel_size=self.ver_w)
        # dense
        self.dense = Dense(embed_dim // 2, activation=activation)
        self.dropout = Dropout(dnn_dropout)
        # norm
        self.use_l2norm = use_l2norm
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_embed = self.user_embedding(tf.reshape(inputs['user'], [-1, ]))  # (None, embed_dim // 2)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), axis=-1)  # (None, seq_len, 1)
        # seq info
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, embed_dim // 2)
        seq_embed *= mask
        # horizontal conv (None, (seq_len - kernel_size + 2 * pad) / stride +1, hor_n)
        hor_info = list()
        for hor_conv in self.hor_conv_list:
            hor_info_i = hor_conv(seq_embed)
            hor_info_i = GlobalMaxPooling1D()(hor_info_i)  # (None, hor_n)
            hor_info.append(hor_info_i)
        hor_info = tf.concat(hor_info, axis=1)
        # vertical conv  (None, (dim - 1 + 2 * pad) / stride + 1, ver_n)
        ver_info = self.ver_conv(tf.transpose(seq_embed, perm=(0, 2, 1)))
        ver_info = tf.reshape(ver_info, shape=(-1, ver_info.shape[1] * ver_info.shape[2]))  # (None, ?)
        # info
        seq_info = self.dense(tf.concat([hor_info, ver_info], axis=-1))  # (None, dim)
        seq_info = self.dropout(seq_info)
        # concat
        user_info = tf.concat([seq_info, user_embed], axis=-1)  # (None, embed_dim)
        # pos info
        pos_info = self.item2_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
        # neg info
        neg_info = self.item2_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # norm
        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            user_info = tf.math.l2_normalize(user_info, axis=-1)
        # scores
        pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info), axis=-1)  # (None, neg_num)
        # loss
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
