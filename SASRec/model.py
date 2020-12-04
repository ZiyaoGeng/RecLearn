"""
Created on Sept 10, 2020

model: Self-Attentive Sequential Recommendation

@author: Ziyao Geng
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, \
    Dropout, Embedding, Input

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
        # self.pos_embedding = Embedding(input_dim=self.maxlen,
        #                                input_length=1,
        #                                output_dim=self.embed_dim,
        #                                mask_zero=False,
        #                                embeddings_initializer='random_uniform',
        #                                embeddings_regularizer=l2(embed_reg))
        # attention block
        self.attention_block = [SelfAttentionBlock(self.d_model, num_heads, ffn_hidden_unit,
                                                   dropout, norm_training, causality) for b in range(blocks)]

    def call(self, inputs):
        # inputs
        seq_inputs, item_inputs = inputs  # (None, maxlen), (None, 1)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)  # (None, maxlen, 1)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        # item info
        item_embed = self.item_embedding(tf.squeeze(item_inputs, axis=-1))  # (None, dim)

        # pos encoding
        pos_encoding = positional_encoding(seq_inputs, self.embed_dim)
        # pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding

        att_outputs = seq_embed  # (None, maxlen, dim)
        att_outputs *= mask

        # self-attention
        for block in self.attention_block:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, dim)
            att_outputs *= mask

        # Here is a difference from the original paper.
        user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
        # predict
        outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(user_info, item_embed), axis=1, keepdims=True))
        return outputs

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        item_inputs = Input(shape=(1,), dtype=tf.int32)
        tf.keras.Model(inputs=[seq_inputs, item_inputs],
                    outputs=self.call([seq_inputs, item_inputs])).summary()


def test_model():
    item_fea_col = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    model = SASRec(item_fea_col, num_heads=8)
    model.summary()


# test_model()