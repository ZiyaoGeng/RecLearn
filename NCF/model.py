"""
Created on Dec 20, 2020

model: Neural Collaborative Filtering

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input

from modules import *


class NCF(Model):
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
        """
        NCF model
        :param feature_columns: A list. user feature columns + item feature columns
        :param hidden_units: A list.
        :param dropout: A scalar.
        :param activation: A string.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NCF, self).__init__(**kwargs)
        if hidden_units is None:
            hidden_units = [64, 32, 16, 8]
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # MF user embedding
        self.mf_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.user_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MF item embedding
        self.mf_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.item_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MLP user embedding
        self.mlp_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.user_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # MLP item embedding
        self.mlp_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.item_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # dnn
        self.dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1), (None, 1/101)
        # user info
        mf_user_embed = self.mf_user_embedding(user_inputs)  # (None, 1, dim)
        mlp_user_embed = self.mlp_user_embedding(user_inputs)  # (None, 1, dim)
        # item
        mf_pos_embed = self.mf_item_embedding(pos_inputs)  # (None, 1, dim)
        mf_neg_embed = self.mf_item_embedding(neg_inputs)  # (None, 1/101, dim)
        mlp_pos_embed = self.mlp_item_embedding(pos_inputs)  # (None, 1, dim)
        mlp_neg_embed = self.mlp_item_embedding(neg_inputs)  # (None, 1/101, dim)
        # MF
        mf_pos_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_pos_embed))  # (None, 1, dim)
        mf_neg_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_neg_embed))  # (None, 1, dim)
        # MLP
        mlp_pos_vector = tf.concat([mlp_user_embed, mlp_pos_embed], axis=-1)  # (None, 1, 2 * dim)
        mlp_neg_vector = tf.concat([tf.tile(mlp_user_embed, multiples=[1, mlp_neg_embed.shape[1], 1]),
                                    mlp_neg_embed], axis=-1)  # (None, 1/101, 2 * dim)
        mlp_pos_vector = self.dnn(mlp_pos_vector)  # (None, 1, dim)
        mlp_neg_vector = self.dnn(mlp_neg_vector)  # (None, 1/101, dim)
        # concat
        pos_vector = tf.concat([mf_pos_vector, mlp_pos_vector], axis=-1)  # (None, 1, 2 * dim)
        neg_vector = tf.concat([mf_neg_vector, mlp_neg_vector], axis=-1)  # (None, 1/101, 2 * dim)
        # pos_vector = mlp_pos_vector
        # neg_vector = mlp_neg_vector
        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)
        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    item_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, item_features]
    model = NCF(features)
    model.summary()


# test_model()
