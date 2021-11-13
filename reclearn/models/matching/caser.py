"""
Created on Nov 18, 2020
Updated on Nov 11, 2021
Reference: "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding", WSDM, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2

from reclearn.models.losses import get_loss


class Caser(Model):
    def __init__(self, fea_cols, hor_n=8, hor_h=2, ver_n=4, activation='relu', dnn_dropout=0.5, loss_name="bpr_loss", gamma=0.5, embed_reg=1e-8, seed=None):
        """
        AttRec
        :param fea_col: A dict contains 'user_num', 'item_num', 'seq_len' and 'embed_dim'.
        :param hor_n: A scalar. The number of horizontal filters.
        :param hor_h: A scalar. Height of horizontal filters.
        :param ver_n: A scalar. The number of vertical filters.
        :param activation: A string. 'relu', 'sigmoid' or 'tanh'.
        :param dnn_dropout: A scalar. The number of dropout.
        :param loss_name: A string. You can specify the current pair-loss function as "bpr_loss" or "hinge_loss".
        :param gamma: A scalar. If hinge_loss is selected as the loss function, you can specify the margin.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param seed: A int scalar.
        """
        super(Caser, self).__init__()
        # user embedding
        self.user_embedding = Embedding(input_dim=fea_cols['user_num'],
                                        input_length=1,
                                        output_dim=fea_cols['embed_dim'] // 2,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=fea_cols['item_num'],
                                        input_length=1,
                                        output_dim=fea_cols['embed_dim'] // 2,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # item2 embedding
        self.item2_embedding = Embedding(input_dim=fea_cols['item_num'],
                                        input_length=1,
                                        output_dim=fea_cols['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # seq_len
        self.seq_len = fea_cols['seq_len']
        # horizontal filters
        self.hor_n = hor_n
        self.hor_h = hor_h if hor_h <= self.seq_len else self.seq_len
        # vertical filters
        self.ver_n = ver_n
        self.ver_w = 1
        # horizontal conv
        self.hor_conv = Conv1D(filters=self.hor_n, kernel_size=self.hor_h)
        # vertical conv, should transpose
        self.ver_conv = Conv1D(filters=self.ver_n, kernel_size=self.ver_w)
        # max_pooling
        self.pooling = GlobalMaxPooling1D()
        # dense
        self.dense = Dense(fea_cols['embed_dim'] // 2, activation=activation)
        self.dropout = Dropout(dnn_dropout)
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_embed = self.user_embedding(inputs['user'])  # (None, embed_dim // 2)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), axis=-1)  # (None, seq_len, 1)
        # seq info
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, embed_dim // 2)
        seq_embed *= mask
        # horizontal conv (None, (seq_len - kernel_size + 2 * pad) / stride +1, hor_n)
        hor_info = self.hor_conv(seq_embed)
        hor_info = self.pooling(hor_info)  # (None, hor_n)
        # vertical conv  (None, (dim - 1 + 2 * pad) / stride + 1, ver_n)
        ver_info = self.ver_conv(tf.transpose(seq_embed, perm=(0, 2, 1)))
        ver_info = tf.reshape(ver_info, shape=(-1, ver_info.shape[1] * ver_info.shape[2]))  # (None, ?)
        # info
        seq_info = self.dense(tf.concat([hor_info, ver_info], axis=-1))  # (None, dim)
        seq_info = self.dropout(seq_info)
        # concat
        user_info = tf.concat([seq_info, user_embed], axis=-1)  # (None, embed_dim)
        # pos info
        pos_info = self.item2_embedding(inputs['pos_item'])  # (None, embed_dim)
        # neg info
        neg_info = self.item2_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # scores
        pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
        pos_scores = tf.tile(pos_scores, [1, neg_info.shape[1]])  # (None, neg_num)
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
