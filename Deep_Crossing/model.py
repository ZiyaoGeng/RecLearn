"""
Created on July 27, 2020

model: Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, ReLU, Layer, Dropout


class Residual_Units(Layer):
    """
    Residual Units
    """
    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: A list. Neural network hidden units.
        :param dim_stack: A scalar. The dimension of inputs unit.
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class Deep_Crossing(keras.Model):
    def __init__(self, feature_columns, hidden_units, res_dropout=0., embed_reg=1e-4):
        """
        Deep&Crossing
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param res_dropout: A scalar. Dropout of resnet.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Deep_Crossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # the total length of embedding layers
        embed_dim = sum([feat['embed_dim'] for feat in self.sparse_feature_columns])
        # the dimension of stack layer
        dim_stack = len(self.dense_feature_columns) + embed_dim
        self.res_network = [Residual_Units(unit, dim_stack) for unit in hidden_units]
        self.res_dropout = Dropout(res_dropout)
        self.dense = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        stack = tf.concat([sparse_embed, dense_inputs], axis=-1)
        r = stack
        for res in self.res_network:
            r = res(r)
        r = self.res_dropout(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()

