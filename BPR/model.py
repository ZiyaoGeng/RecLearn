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
    def __init__(self, feature_columns, mode='inner', embed_reg=1e-6):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :mode: A string. 'inner' or 'dist'.
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
        super(BPR, self).__init__()
        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # mode
        self.mode = mode
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

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs  # (None, 1), (None, 1)
        # user info
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        # item
        pos_embed = self.item_embedding(tf.squeeze(pos_inputs, axis=-1))  # (None, dim)
        neg_embed = self.item_embedding(tf.squeeze(neg_inputs, axis=-1))  # (None, dim)
        if self.mode == 'inner':
            # calculate positive item scores and negative item scores
            pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=1, keepdims=True)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(user_embed, neg_embed), axis=1, keepdims=True)  # (None, 1)
            # add loss. Computes softplus: log(exp(features) + 1)
            # self.add_loss(tf.reduce_mean(tf.math.softplus(neg_scores - pos_scores)))
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            pos_scores = tf.reduce_sum(tf.square(user_embed - pos_embed), axis=-1, keepdims=True)
            neg_scores = tf.reduce_sum(tf.square(user_embed - neg_embed), axis=-1, keepdims=True)
            self.add_loss(tf.reduce_sum(tf.maximum(neg_scores - pos_scores + 0.5, 0)))
        return pos_scores, neg_scores

    def summary(self):
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
            outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
    item_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    features = [user_features, item_features]
    model = BPR(features)
    model.summary()


# test_model()