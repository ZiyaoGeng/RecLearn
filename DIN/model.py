"""
Created on May 23, 2020

model: Deep interest network for click-through rate prediction

@author: Ziyao Geng
"""
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout
from tensorflow.keras.regularizers import l2

from modules import *


class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40), 
        ffn_hidden_units=(80, 40), att_activation='sigmoid', ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_reg=1e-4):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DIN, self).__init__()
        self.maxlen = maxlen

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.seq_len = len(behavior_feature_list)

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] in behavior_feature_list]

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)
        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice())\
             for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        # dense_inputs and sparse_inputs is empty
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        # other
        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq, item embedding and category embedding should concatenate
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, i]) for i in range(self.seq_len)], axis=-1)
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.seq_len)], axis=-1)
    
        # att
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed])  # (None, d * 2)

        # concat user_info(att hist), cadidate item embedding, other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)

        info_all = self.bn(info_all)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len, ), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len, ), dtype=tf.int32)
        seq_inputs = Input(shape=(self.seq_len, self.maxlen), dtype=tf.int32)
        item_inputs = Input(shape=(self.seq_len, ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()


def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DIN(features, behavior_list)
    model.summary()


# test_model()