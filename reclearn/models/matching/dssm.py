"""
Created on Mar 31, 2022
Reference: "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data", CIKM, 2013
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2
from reclearn.layers import MLP
from reclearn.models.losses import get_loss


class DSSM(Model):
    def __init__(self, user_num, item_num, embed_dim, user_mlp, item_mlp, activation='relu',
                 dnn_dropout=0., use_l2norm=False, loss_name="binary_cross_entropy_loss",
                 gamma=0.5, embed_reg=0., seed=None):
        """DSSM: The two-tower matching model commonly used in industry.
        Args:
            :param user_num: An integer type. The largest user index + 1.
            :param item_num: An integer type. The largest item index + 1.
            :param embed_dim: An integer type. Embedding dimension of user vector and item vector.
            :param user_mlp: A list of user MLP hidden units such as [128, 64, 32].
            :param item_mlp: A list of item MLP hidden units such as [128, 64, 32] and
            the last unit must be equal to the user's.
            :param activation: A string. Activation function name of user and item MLP layer.
            :param dnn_dropout: Float between 0 and 1. Dropout of user and item MLP layer.
            :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
            :param loss_name: A string. You can specify the current point-loss function 'binary_cross_entropy_loss' or
            pair-loss function as 'bpr_loss'、'hinge_loss'.
            :param gamma: A scalar. If hinge_loss is selected as the loss function, you can specify the margin.
            :param embed_reg: A float type. The regularizer of embedding.
            :param seed: A Python integer to use as random seed.
        :return:
        """
        super(DSSM, self).__init__()
        if user_mlp[-1] != item_mlp[-1]:
            raise ValueError("The last value of user_mlp must be equal to item_mlp's.")
        # user embedding
        self.user_embedding_table = Embedding(input_dim=user_num,
                                              input_length=1,
                                              output_dim=embed_dim,
                                              embeddings_initializer='random_normal',
                                              embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding_table = Embedding(input_dim=item_num,
                                              input_length=1,
                                              output_dim=embed_dim,
                                              embeddings_initializer='random_normal',
                                              embeddings_regularizer=l2(embed_reg))
        # user_mlp_layer
        self.user_mlp_layer = MLP(user_mlp, activation, dnn_dropout)
        # item_mlp_layer
        self.item_mlp_layer = MLP(item_mlp, activation, dnn_dropout)
        self.use_l2norm = use_l2norm
        self.loss_name = loss_name
        self.gamma = gamma
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs):
        # user info
        user_info = self.user_embedding_table(inputs['user'])  # (None, embed_dim)
        # item info
        pos_info = self.item_embedding_table(inputs['pos_item'])  # (None, embed_dim)
        neg_info = self.item_embedding_table(inputs['neg_item'])  # (None, neg_num, embed_dim)
        # mlp
        user_info = self.user_mlp_layer(user_info)
        pos_info = self.item_mlp_layer(pos_info)
        neg_info = self.item_mlp_layer(neg_info)
        # norm
        if self.use_l2norm:
            user_info = tf.math.l2_normalize(user_info, axis=-1)
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
        # calculate similar scores.
        pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info), axis=-1)  # (None, neg_num)
        # add loss
        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()