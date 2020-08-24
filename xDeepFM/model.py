"""
Created on August 20, 2020

model: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Flatten, Dense, Input


class DNN(layers.Layer):
    """
    DNN part
    """
    def __init__(self, hidden_units, dnn_dropout, dnn_activation):
        """

        :param hidden_units: list of hidden layer units's numbers
        :param dnn_dropout: dropout number
        :param dnn_activation: activation function
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=dnn_activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class Linear(layers.Layer):
    """
    Linear Part
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result


class CIN(layers.Layer):
    """
    CIN part
    """
    def __init__(self, cin_size, l2_reg=1e-4):
        """

        :param cin_size: [H_1, H_2 ,..., H_k], a list of the number of layers
        :param l2_reg:
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.embedding_nums = input_shape[1]
        self.field_nums = list(self.cin_size)
        self.field_nums.insert(0, self.embedding_nums)
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)

            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            result_3 = tf.transpose(result_2, perm=[1, 0, 2])

            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')

            result_5 = tf.transpose(result_4, perm=[0, 2, 1])

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result,  axis=-1)

        return result


class xDeepFM(keras.Model):
    def __init__(self, feature_columns, hidden_units=(200, 200), cin_size=(128, 128,), dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-5, cin_reg=1e-5):
        """
        xDeepFM architecture
        :param feature_columns: a list containing dense and sparse column feature information
        :param hidden_units: a list of dnn hidden units
        :param cin_size: a list of the number of CIN layers
        :param dnn_dropout: dropout of dnn
        :param dnn_activation: activation function of dnn
        :param embed_reg: the regularizer of embedding
        :param cin_reg: the regularizer of cin
        """
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.linear = Linear()
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        self.dense = Dense(1)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        # linear
        linear_out = self.linear(sparse_inputs)

        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # cin
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)
        # dnn
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)

        total_input = tf.concat([linear_out, cin_out, dnn_out, dense_inputs], axis=-1)
        output = tf.nn.sigmoid(self.dense(total_input))
        return output

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()
