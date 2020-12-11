"""
Created on Nov 22, 2020

model: Sequential Recommender System---DNN

@author: Ziyao Geng
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Embedding, Input


class DNN(tf.keras.Model):
    def __init__(self, item_fea_col, maxlen=40, hidden_units=128, activation='relu', embed_reg=1e-6):
        """
        DNN model
        :param item_fea_col: A dict contains 'feat_name', 'feat_num' and 'embed_dim'.
        :param maxlen: A scalar. Number of length of sequence.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DNN, self).__init__()
        # maxlen
        self.maxlen = maxlen
        self.item_fea_col = item_fea_col
        # item embed
        embed_dim = self.item_fea_col['embed_dim']
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.dense = Dense(hidden_units, activation=activation)
        self.dense2 = Dense(embed_dim, activation=activation)

    def call(self, inputs):
        seq_inputs, item_inputs = inputs
        # mask
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)  # (None, maxlen)
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        seq_embed *= tf.expand_dims(mask, axis=-1)
        seq_embed_mean = tf.reduce_mean(seq_embed, axis=1)  # (None, dim)
        user_embed = self.dense(seq_embed_mean)
        user_embed = self.dense2(user_embed)
        # item info
        item_embed = self.item_embedding(tf.squeeze(item_inputs, axis=-1))  # (None, dim)
        # output
        outputs = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(user_embed, item_embed), axis=1, keepdims=True))
        return outputs

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        item_inputs = Input(shape=(1,), dtype=tf.int32)
        tf.keras.Model(inputs=[seq_inputs, item_inputs],
                       outputs=self.call([seq_inputs, item_inputs])).summary()


def test_model():
    item_fea_col = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    model = DNN(item_fea_col)
    model.summary()


# test_model()