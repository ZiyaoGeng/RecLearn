"""
Created on August 3, 2020

model: Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input


class AFM(keras.Model):
    def __init__(self, feature_columns, mode, activation='relu', embed_reg=1e-4):
        """
        AFM 
        :param feature_columns: A list. dense_feature_columns and sparse_feature_columns
        :param mode:A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param activation: A string. Activation function of attention.
        :param embed_reg: A scalar. the regularizer of embedding
        """
        super(AFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        if self.mode == 'att':
            t = (len(self.embed_layers) - 1) * len(self.embed_layers) // 2
            self.attention_W = Dense(units=t, activation=activation)
            self.attention_dense = Dense(units=1, activation=None)

        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        # Input Layer
        dense_inputs, sparse_inputs = inputs
        # Embedding Layer 
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), perm=[1, 0, 2])  # (None, len(sparse_inputs), embed_dim)
        # Pair-wise Interaction Layer
        # for loop is badly
        element_wise_product_list = []
        # t = (len - 1) * len /2, k = embed_dim
        for i in range(embed.shape[1]):
            for j in range(i+1, embed.shape[1]):
                element_wise_product_list.append(tf.multiply(embed[:, i], embed[:, j]))
        element_wise_product = tf.transpose(tf.stack(element_wise_product_list), [1, 0, 2])  # (None, t, k)
        # mode
        if self.mode == 'max':
            # MaxPooling Layer
            x = tf.reduce_sum(element_wise_product, axis=1)   # (None, k)
        elif self.mode == 'avg':
            # AvgPooling Layer
            x = tf.reduce_mean(element_wise_product, axis=1)  # (None, k)
        else:
            # Attention Layer
            x = self.attention(element_wise_product)  # (None, k) 
        # Output Layer
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()

    def attention(self, keys):
        a = self.attention_W(keys)  # (None, t, t)
        a = self.attention_dense(a)  # (None, t, 1) 
        a_score = tf.nn.softmax(a)  # (None, t, 1)
        a_score = tf.transpose(a_score, [0, 2, 1])  # (None, 1, t)
        outputs = tf.reshape(tf.matmul(a_score, keys), shape=(-1, keys.shape[2]))  # (None, k)
        return outputs
