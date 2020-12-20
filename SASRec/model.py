"""
Updated on Dec 20, 2020

model: Self-Attentive Sequential Recommendation

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Input

from modules import *


class SASRec(tf.keras.Model):
    def __init__(self, item_fea_col, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=40, norm_training=True, causality=False, embed_reg=1e-6):
        """
        SASRec model
        :param item_fea_col: A dict contains 'feat_name', 'feat_num' and 'embed_dim'.
        :param blocks: A scalar. The Number of blocks.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param maxlen: A scalar. Number of length of sequence
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        :param embed_reg: A scalar. The regularizer of embedding
        """
        super(SASRec, self).__init__()
        # sequence length
        self.maxlen = maxlen
        # item feature columns
        self.item_fea_col = item_fea_col
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        # d_model must be the same as embedding_dim, because of residual connection
        self.d_model = self.embed_dim
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        self.pos_embedding = Embedding(input_dim=self.maxlen,
                                       input_length=1,
                                       output_dim=self.embed_dim,
                                       mask_zero=False,
                                       embeddings_initializer='random_uniform',
                                       embeddings_regularizer=l2(embed_reg))
        self.dropout = Dropout(dropout)
        # attention block
        self.encoder_layer = [EncoderLayer(self.d_model, num_heads, ffn_hidden_unit,
                                           dropout, norm_training, causality) for b in range(blocks)]

    def call(self, inputs, training=None):
        # inputs
        seq_inputs, pos_inputs, neg_inputs = inputs  # (None, maxlen), (None, 1), (None, 1)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)  # (None, maxlen, 1)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # pos encoding
        # pos_encoding = positional_encoding(seq_inputs, self.embed_dim)
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding
        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed  # (None, maxlen, dim)
        att_outputs *= mask

        # self-attention
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, dim)
            att_outputs *= mask

        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
        user_info = tf.expand_dims(att_outputs[:, -1], axis=1)  # (None, 1, dim)
        # item info
        pos_info = self.item_embedding(pos_inputs)  # (None, 1, dim)
        neg_info = self.item_embedding(neg_inputs)  # (None, 1/100, dim)
        pos_logits = tf.reduce_sum(user_info * pos_info, axis=-1)  # (None, 1)
        neg_logits = tf.reduce_sum(user_info * neg_info, axis=-1)  # (None, 1)
        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        tf.keras.Model(inputs=[seq_inputs, pos_inputs, neg_inputs],
                       outputs=self.call([seq_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    item_fea_col = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    model = SASRec(item_fea_col, num_heads=8)
    model.summary()


# test_model()