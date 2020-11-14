"""
Created on Nov 13, 2020

model: BPR: Bayesian Personalized Ranking from Implicit Feedback

@author: Ziyao Geng
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2


class BPR(Model):
    def __init__(self, feature_columns, embed_reg=1e-6):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
        super(BPR, self).__init__()
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # field num
        self.user_field_num = len(self.user_fea_col)
        self.item_field_num = len(self.item_fea_col)
        # user embedding layers [id, age,...]
        self.embed_user_layers = [Embedding(input_dim=feat['feat_num'],
                                            input_length=1,
                                            output_dim=feat['embed_dim'],
                                            embeddings_initializer='random_uniform',
                                            embeddings_regularizer=l2(embed_reg))
                                  for feat in self.user_fea_col]
        # item embedding layers [id, cate_id, ...]
        self.embed_item_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.item_fea_col]

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, user_field_num), (None, item_field_num)
        # user info
        user_embed = tf.add_n([self.embed_user_layers[i](user_inputs[:, i])
                               for i in range(self.user_field_num)]) / self.user_field_num  # (None, dim)
        # item  info
        pos_embed = tf.add_n([self.embed_item_layers[i](pos_inputs[:, i])
                              for i in range(self.item_field_num)]) / self.item_field_num  # (None, dim)
        neg_embed = tf.add_n([self.embed_item_layers[i](neg_inputs[:, i])
                              for i in range(self.item_field_num)]) / self.item_field_num  # (None, dim)
        # calculate positive item scores and negative item scores
        pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(user_embed, neg_embed), axis=1, keepdims=True)  # (None, 1)
        # add loss. Computes softplus: log(exp(features) + 1)
        # self.add_loss(tf.reduce_mean(tf.math.softplus(neg_scores - pos_scores)))
        self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        return pos_scores, neg_scores

    def summary(self):
        user_inputs = Input(shape=(self.user_field_num, ), dtype=tf.int32)
        pos_inputs = Input(shape=(self.item_field_num, ), dtype=tf.int32)
        neg_inputs = Input(shape=(self.item_field_num, ), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
            outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    user_features = [{'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}]
    item_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                    {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8}]
    features = [user_features, item_features]
    model = BPR(features)
    model.summary()


# test_model()