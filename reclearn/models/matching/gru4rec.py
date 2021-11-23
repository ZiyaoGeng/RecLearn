"""
Created on Nov 20, 2021
Reference: "Session-based Recommendation with Recurrent Neural Networks", ICLR, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, GRU
from tensorflow.keras.regularizers import l2

from reclearn.models.losses import get_loss


class GRU4Rec(Model):
    def __init__(self, feature_columns, seq_len=40, gru_layers=1, gru_unit=64, gru_activation='tanh',
                 dnn_dropout=0., loss_name='bpr_loss', gamma=0.5, embed_reg=0., seed=None):
        """GRU4Rec
        Args:
            :param feature_columns:  A dict containing
            {'user': {'feat_name':, 'feat_num':, 'embed_dim'}, 'item': {...}, ...}.
            :param seq_len: A scalar. The length of the input sequence.
            :param gru_layers: A scalar. The number of GRU Layers.
            :param gru_unit: A scalar. The unit of GRU Layer.
            :param gru_activation: A string. The name of activation function. Default 'tanh'.
            :param dnn_dropout: A scalar. Number of dropout.
            :param loss_name: A string. You can specify the current pair-loss function as "bpr_loss" or "hinge_loss".
            :param gamma: A scalar. If hinge_loss is selected as the loss function, you can specify the margin.
            :param embed_reg: A scalar. The regularizer of embedding.
            :param seed: A int scalar.
        :return:
        """
        super(GRU4Rec, self).__init__()
        self.item_embedding = Embedding(input_dim=feature_columns['item']['feat_num'],
                                        input_length=1,
                                        output_dim=feature_columns['item']['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.dropout = Dropout(dnn_dropout)
        self.gru_layers = [
            GRU(units=gru_unit, activation=gru_activation, return_sequences=True)
            if i < gru_layers - 1 else
            GRU(units=gru_unit, activation=gru_activation, return_sequences=False)
            for i in range(gru_layers)
        ]
        self.dense = Dense(units=feature_columns['item']['embed_dim'])
        # loss name
        self.loss_name = loss_name
        self.gamma = gamma
        # seq_len
        self.seq_len = seq_len
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
        pos_info = self.item_embedding(inputs['pos_item'])  # (None, embed_dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # norm
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
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()