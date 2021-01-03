"""
Created on July 20, 2020

model: Product-based Neural Networks for User Response Prediction

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Layer, Dropout


class DNN(Layer):
    """
	Deep Neural Network
	"""

    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
		"""
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class PNN(keras.Model):
    def __init__(self, feature_columns, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=1e-4, w_z_reg=1e-4, w_p_reg=1e-4, l_b_reg=1e-4):
        """
        Product-based Neural Networks
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param mode: A string. 'in' IPNN or 'out'OPNN.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param w_z_reg: A scalar. The regularizer of w_z_ in product layer
        :param w_p_reg: A scalar. The regularizer of w_p in product layer
        :param l_b_reg: A scalar. The regularizer of l_b in product layer
        """
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # the number of feature fields
        self.field_num = len(self.sparse_feature_columns)
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # parameters
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform',
                                   regularizer=l2(w_z_reg),
                                   trainable=True
                                   )
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(len(self.sparse_feature_columns)*( len(self.sparse_feature_columns)-1  )//2, self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       reguarizer=l2(w_p_reg),
                                       trainable=True)
        # out
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(len(self.sparse_feature_columns)*( len(self.sparse_feature_columns)-1  )//2,self.embed_dim, self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        self.l_b = self.add_weight(name='l_b', shape=(hidden_units[0], ),
                                   initializer='random_uniform',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        # dnn
        self.dnn_network = DNN(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])  # (None, field_num, embed_dim)
        l_z = tf.tensordot(embed, self.w_z, axes=2)  # (None, hidden[0])
        # product layer
        row = []
        col = []
        for i in range(len(self.sparse_feature_columns) - 1):
            for j in range(i + 1, len(self.sparse_feature_columns)):
                row.append(i)
                col.append(j)
        p = tf.gather(embed,row,axis=1)
        q = tf.gather(embed,col, axis=1)
        # product layer
        if self.mode == 'in':
            l_p = tf.tensordot( p*q, self.w_p, axes=2)  # (None, hidden[0])
        else:  # out
            u = tf.expand_dims(q, 2)  # (None, field_num(field_num-1)/2, 1, emb_num)
            v = tf.expand_dims(p, 2)  # (None, field_num(field_num-1)/2, 1, emb_num)
            l_p = tf.tensordot(tf.matmul(tf.transpose(u, [0, 1, 3, 2]), v), self.w_p, axes=3) # (None, hidden[0])

        l_1 = tf.nn.relu(tf.concat([l_z+l_p+self.l_b, dense_inputs], axis=-1))
        # dnn layer
        dnn_x = self.dnn_network(l_1)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()
