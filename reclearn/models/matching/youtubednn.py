"""
Created on Mar 31, 2022
Reference: "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data", CIKM, 2013
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from reclearn.layers import MLP


class YoutubeDNN(Model):
    def __init__(self, item_num, embed_dim, user_mlp, activation='relu',
                 dnn_dropout=0., use_l2norm=False, neg_num=4, batch_size=512,
                 embed_reg=0., seed=None):
        """YouTubeDNN: Select this section of the two-towers recall model.
        Args:
            :param item_num: An integer type. The largest item index + 1.
            :param embed_dim: An integer type. Embedding dimension of item vector.
            :param user_mlp: A list of user MLP hidden units such as [128, 64, 32].
            User initial vector is the mean of the user's historical behavior sequence vector.
            :param activation: A string. Activation function name of user and item MLP layer.
            :param dnn_dropout: Float between 0 and 1. Dropout of user and item MLP layer.
            :param use_l2norm: A boolean. Whether user embedding, item embedding should be normalized or not.
            :param neg_num: A integer type. The number of negative samples for each positive sample.
            :param batch_size: A integer type. The number of samples per batch.
            :param embed_reg: A float type. The regularizer of embedding.
            :param seed: A Python integer to use as random seed.
        :return:
        """
        super(YoutubeDNN, self).__init__()
        with tf.name_scope("Embedding_layer"):
            # item embedding
            self.item_embedding_table = self.add_weight(name='item_embedding_table',
                                                        shape=(item_num, embed_dim),
                                                        initializer='random_normal',
                                                        regularizer=l2(embed_reg),
                                                        trainable=True)
            # embedding bias
            self.embedding_bias = self.add_weight(name='embedding_bias',
                                                  shape=(item_num,),
                                                  initializer=tf.zeros_initializer(),
                                                  trainable=False)
        # user_mlp_layer
        self.user_mlp_layer = MLP(user_mlp, activation, dnn_dropout)
        self.use_l2norm = use_l2norm
        self.embed_dim = embed_dim
        self.item_num = item_num
        self.neg_num = neg_num
        self.batch_size = batch_size
        # seed
        tf.random.set_seed(seed)

    def call(self, inputs, training=False):
        # user info
        user_info = tf.reduce_mean(tf.nn.embedding_lookup(self.item_embedding_table, inputs['click_seq']), axis=1)
        # mlp
        user_info = self.user_mlp_layer(user_info)
        if user_info.shape[-1] != self.embed_dim:
            raise ValueError("The last hidden unit must be equal to the embedding dimension.")
        # norm
        if self.use_l2norm:
            user_info = tf.math.l2_normalize(user_info, axis=-1)
        if training:
            # train, sample softmax loss
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=self.item_embedding_table,
                biases=self.embedding_bias,
                labels=tf.reshape(inputs['pos_item'], shape=[-1, 1]),
                inputs=user_info,
                num_sampled=self.neg_num * self.batch_size,
                num_classes=self.item_num
            ))
            # add loss
            self.add_loss(loss)
            return loss
        else:
            # predict/eval
            pos_info = tf.nn.embedding_lookup(self.item_embedding_table, inputs['pos_item'])  # (None, embed_dim)
            neg_info = tf.nn.embedding_lookup(self.item_embedding_table, inputs['neg_item'])  # (None, neg_num, embed_dim)
            # calculate similar scores.
            pos_scores = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1, keepdims=True)  # (None, 1)
            neg_scores = tf.reduce_sum(tf.multiply(tf.expand_dims(user_info, axis=1), neg_info), axis=-1)  # (None, neg_num)
            logits = tf.concat([pos_scores, neg_scores], axis=-1)
            return logits

    def summary(self):
        inputs = {
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)  # suppose neg_num=1
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()