"""
Created on Nov 18, 2020

model: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding

@author: Ziyao Geng
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout


class Caser(Model):
    def __init__(self, feature_columns, maxlen=40, hor_n=2, hor_h=8, ver_n=8, dropout=0.5, activation='relu', embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param hor_n: A scalar. The number of horizontal filters.
        :param hor_h: A scalar. Height of horizontal filters.
        :param ver_n: A scalar. The number of vertical filters.
        :param activation: A string. 'relu', 'sigmoid' or 'tanh'.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Caser, self).__init__()
        # maxlen
        self.maxlen = maxlen
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # total number of item set
        self.total_item = self.item_fea_col['feat_num']
        # horizontal filters
        self.hor_n = hor_n
        self.hor_h = hor_h if hor_h <= self.maxlen else self.maxlen
        # vertical filters
        self.ver_n = ver_n
        self.ver_w = 1
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
        # horizontal conv
        self.hor_conv = Conv1D(filters=self.hor_n, kernel_size=self.hor_h)
        # vertical conv, should transpose
        self.ver_conv = Conv1D(filters=self.ver_n, kernel_size=self.ver_w)
        # max_pooling
        self.pooling = GlobalMaxPooling1D()
        # dense
        self.dense = Dense(self.embed_dim, activation=activation)
        self.dropout = Dropout(dropout)
        self.dense_final = Dense(self.total_item, activation=None)

    def call(self, inputs):
        # input
        user_inputs, seq_inputs = inputs
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # horizontal conv (None, (maxlen - kernel_size + 2 * pad) / stride +1, hor_n)
        hor_info = self.hor_conv(seq_embed)
        hor_info = self.pooling(hor_info)  # (None, hor_n)
        # vertical conv  (None, (dim - 1 + 2 * pad) / stride + 1, ver_n)
        ver_info = self.ver_conv(tf.transpose(seq_embed, perm=(0, 2, 1)))
        ver_info = tf.reshape(ver_info, shape=(-1, ver_info.shape[1] * ver_info.shape[2]))  # (None, ?)
        # info
        seq_info = self.dense(tf.concat([hor_info, ver_info], axis=-1))  # (None, d)
        seq_info = self.dropout(seq_info)
        # concat
        info = tf.concat([seq_info, user_embed], axis=-1)  # (None, 2 * d)
        # pred
        pred_y = tf.nn.sigmoid(self.dense_final(info))  # (None, total_num)
        return pred_y

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs],
              outputs=self.call([user_inputs, seq_inputs])).summary()


def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    seq_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, seq_features]
    model = Caser(features)
    model.summary()


# test_model()

