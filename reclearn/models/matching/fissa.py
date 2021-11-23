"""
Created on Nov 20, 2021
Reference: "FISSA: fusing item similarity models with self-attention networks for sequential recommendation",
            RecSys, 2020
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, LayerNormalization, Dropout
from tensorflow.keras.regularizers import l2

from reclearn.layers import TransformerEncoder, LBA, Item_similarity_gating
from reclearn.models.losses import get_loss


class FISSA(Model):
    def __init__(self, feature_columns, seq_len=40, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dnn_dropout=0., layer_norm_eps=1e-6, loss_name="bpr_loss", gamma=0.5, embed_reg=0., seed=None):
        """FISSA
        Args:
            :param feature_columns:  A dict containing
                {'user': {'feat_name':, 'feat_num':, 'embed_dim'}, 'item': {...}, ...}.
            :param seq_len: A scalar. The length of the input sequence.
            :param mode: A string. inner or dist.
            :param w: A scalar. The weight of short interest.
            :param loss_name: A string. You can specify the current pair-loss function as "bpr_loss" or "hinge_loss".
            :param gamma: A scalar. If hinge_loss is selected as the loss function, you can specify the margin.
            :param embed_reg: A scalar. The regularizer of embedding.
            :param seed: A int scalar.
        """
        super(FISSA, self).__init__()
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
        # item2 embedding, not share embedding
        self.item2_embedding = Embedding(input_dim=feature_columns['item']['feat_num'],
                                        input_length=1,
                                        output_dim=feature_columns['item']['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        # encoder
        self.encoder_layer = [TransformerEncoder(feature_columns['item']['embed_dim'], num_heads, ffn_hidden_unit,
                                                 dnn_dropout, layer_norm_eps) for _ in range(blocks)]
        self.lba = LBA(dnn_dropout)
        self.gating = Item_similarity_gating(dnn_dropout)
        # layer normalization
        self.layer_norm = LayerNormalization()
        # dropout
        self.dropout = Dropout(dnn_dropout)
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
        seq_embed = self.layer_norm(seq_embed)
        att_outputs = seq_embed  # (None, seq_len, embed_dim)
        att_outputs *= mask
        # transformer encoder part
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, embed_dim)
            att_outputs *= mask

        local_info = tf.slice(att_outputs, begin=[0, self.seq_len - 1, 0], size=[-1, 1, -1])  # (None, 1, embed_dim)
        global_info = self.lba([seq_embed, seq_embed, mask])  # (None, embed_dim)

        pos_info = self.item_embedding(inputs['pos_item'])  # (None, dim)
        neg_info = self.item_embedding(inputs['neg_item'])  # (None, neg_num, dim)

        weights = self.gating([tf.tile(tf.slice(seq_embed, begin=[0, self.seq_len - 1, 0], size=[-1, 1, -1]), [1, neg_info.shape[1] + 1, 1]),
                                tf.tile(local_info, [1, neg_info.shape[1] + 1, 1]),
                                tf.concat([tf.expand_dims(pos_info, axis=1), neg_info], 1)]
                              )  # (None, 1 + neg_num, 1)

        user_info = tf.multiply(local_info, weights) + \
                    tf.multiply(tf.expand_dims(global_info, axis=1), tf.ones_like(weights) - weights)  # (None, 1 + neg_num, embed_dim)
        # norm
        pos_info = tf.math.l2_normalize(pos_info, axis=-1)
        neg_info = tf.math.l2_normalize(neg_info, axis=-1)
        user_info = tf.math.l2_normalize(user_info, axis=-1)

        pos_scores = tf.reduce_sum(tf.multiply(tf.slice(user_info, [0, 0, 0], [-1, 1, -1]), tf.expand_dims(pos_info, axis=1)), axis=-1)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(tf.slice(user_info, [0, 1, 0], [-1, neg_info.shape[1], -1]), neg_info), axis=-1)  # (None, neg_num)
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