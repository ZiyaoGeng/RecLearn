"""
Created on Dec 20, 2020
Updated on Nov 20, 2021
Reference: "Self-Attentive Sequential Recommendation", ICDM, 2018
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import TransformerEncoder
from reclearn.models.losses import get_loss


class SASRec(Model):
    def __init__(self, feature_columns, seq_len=40, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dnn_dropout=0., layer_norm_eps=1e-6, use_l2norm=False, loss_name="bpr_loss", gamma=0.5, embed_reg=0., seed=None):
        """Self-Attentive Sequential Recommendation
        :param feature_columns:  A dict containing
                {'user': {'feat_name':, 'feat_num':, 'embed_dim'}, 'item': {...}, ...}.
        :param seq_len: A scalar. The length of the input sequence.
        :param blocks: A scalar. The Number of blocks.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dnn_dropout: A scalar. Number of dropout.
        :param layer_norm_eps: A scalar. Small float added to variance to avoid dividing by zero.
        :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
        :param loss_name: A string. You can specify the current pair-loss function as "bpr_loss" or "hinge_loss".
        :param gamma: A scalar. If hinge_loss is selected as the loss function, you can specify the margin.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param seed: A int scalar.
        """
        super(SASRec, self).__init__()
        # item embedding
        self.item_embedding = Embedding(input_dim=feature_columns['item']['feat_num'],
                                        input_length=1,
                                        output_dim=feature_columns['item']['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.pos_embedding = Embedding(input_dim=seq_len,
                                       input_length=1,
                                       output_dim=feature_columns['item']['embed_dim'],
                                       embeddings_initializer='random_normal',
                                       embeddings_regularizer=l2(embed_reg))
        self.dropout = Dropout(dnn_dropout)
        # multi encoder block
        self.encoder_layer = [TransformerEncoder(feature_columns['item']['embed_dim'], num_heads, ffn_hidden_unit,
                                                 dnn_dropout, layer_norm_eps) for _ in range(blocks)]
        # norm
        self.use_l2norm = use_l2norm
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
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), axis=-1)  # (None, seq_len, 1)
        # pos encoding
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.seq_len)), axis=0)  # (1, seq_len, embed_dim)
        seq_embed += pos_encoding  # (None, seq_len, embed_dim), broadcasting
        
        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed  # (None, seq_len, embed_dim)
        att_outputs *= mask
        # transformer encoder part
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, embed_dim)
            att_outputs *= mask
        # user_info. There are two ways to get the user vector.
        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
        user_info = tf.slice(att_outputs, begin=[0, self.seq_len-1, 0], size=[-1, 1, -1])  # (None, 1, embed_dim)
        # item info contain pos_info and neg_info.
        pos_info = self.item_embedding(inputs['pos_item'])  # (None, dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, dim)
        # norm
        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            user_info = tf.math.l2_normalize(user_info, axis=-1)
        pos_scores = tf.reduce_sum(tf.multiply(user_info, tf.expand_dims(pos_info, axis=1)), axis=-1)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(user_info, neg_info), axis=-1)  # (None, neg_num)
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
