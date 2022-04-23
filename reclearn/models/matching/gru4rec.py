"""
Created on Nov 20, 2021
Updated on Apr 23, 2022
Reference: "Session-based Recommendation with Recurrent Neural Networks", ICLR, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, GRU
from tensorflow.keras.regularizers import l2

from reclearn.models.losses import get_loss


class GRU4Rec(Model):
    def __init__(self, item_num, embed_dim, gru_layers=1, gru_unit=64, gru_activation='tanh',
                 dnn_dropout=0., use_l2norm=False, loss_name='bpr_loss', gamma=0.5, embed_reg=0., seed=None):
        """GRU4Rec, Sequential Recommendation Model.
        Args:
            :param item_num: An integer type. The largest item index + 1.
            :param embed_dim: An integer type. Embedding dimension of item vector.
            :param gru_layers: An integer type. The number of GRU Layers.
            :param gru_unit:An integer type. The unit of GRU Layer.
            :param gru_activation: A string. The name of activation function. Default 'tanh'.
            :param dnn_dropout: Float between 0 and 1. Dropout of user and item MLP layer.
            :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
            :param loss_name: A string. You can specify the current point-loss function 'binary_cross_entropy_loss' or
            pair-loss function as 'bpr_loss'、'hinge_loss'.
            :param gamma: A float type. If hinge_loss is selected as the loss function, you can specify the margin.
            :param embed_reg: A float type. The regularizer of embedding.
            :param seed: A Python integer to use as random seed.
        :return:
        """
        super(GRU4Rec, self).__init__()
        self.item_embedding = Embedding(input_dim=item_num,
                                        input_length=1,
                                        output_dim=embed_dim,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.dropout = Dropout(dnn_dropout)
        self.gru_layers = [
            GRU(units=gru_unit, activation=gru_activation, return_sequences=True)
            if i < gru_layers - 1 else
            GRU(units=gru_unit, activation=gru_activation, return_sequences=False)
            for i in range(gru_layers)
        ]
        self.dense = Dense(units=embed_dim)
        # norm
        self.use_l2norm = use_l2norm
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # seq info
        seq_embed = self.item_embedding(inputs['click_seq'])  # (None, seq_len, dim)
        # mask
        mask = tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32)  # (None, seq_len)
        seq_embed = tf.multiply(seq_embed, tf.expand_dims(mask, axis=-1))
        # dropout
        seq_info = self.dropout(seq_embed)
        # gru
        for gru_layer in self.gru_layers:
            seq_info = gru_layer(seq_info)
        seq_info = self.dense(seq_info)
        # positive, negative embedding vector.
        pos_info = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))  # (None, embed_dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # norm
        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            seq_info = tf.math.l2_normalize(seq_info, axis=-1)
        # calculate positive item scores and negative item scores
        pos_scores = tf.reduce_sum(tf.multiply(seq_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(seq_info, axis=1), neg_info), axis=-1)  # (None, neg_num)
        # loss
        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        inputs = {
            'click_seq': Input(shape=(100,), dtype=tf.int32),  # suppose sequence length=1
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()