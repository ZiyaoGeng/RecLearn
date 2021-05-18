"""
Created on July 31, 2020
Updated on May 18, 2021

model: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer

from modules import *


class DeepFM(Model):
	def __init__(self, feature_columns, hidden_units=(200, 200, 200), dnn_dropout=0.,
				 activation='relu', fm_w_reg=1e-6, embed_reg=1e-6):
		"""
		DeepFM
		:param feature_columns: A list. sparse column feature information.
		:param hidden_units: A list. A list of dnn hidden units.
		:param dnn_dropout: A scalar. Dropout of dnn.
		:param activation: A string. Activation function of dnn.
		:param fm_w_reg: A scalar. The regularizer of w in fm.
		:param embed_reg: A scalar. The regularizer of embedding.
		"""
		super(DeepFM, self).__init__()
		self.sparse_feature_columns = feature_columns
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
										 input_length=1,
										 output_dim=feat['embed_dim'],
										 embeddings_initializer='random_normal',
										 embeddings_regularizer=l2(embed_reg))
			for i, feat in enumerate(self.sparse_feature_columns)
		}
		self.index_mapping = []
		self.feature_length = 0
		for feat in self.sparse_feature_columns:
			self.index_mapping.append(self.feature_length)
			self.feature_length += feat['feat_num']
		self.embed_dim = self.sparse_feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim
		self.fm = FM(self.feature_length, fm_w_reg)
		self.dnn = DNN(hidden_units, activation, dnn_dropout)
		self.dense = Dense(1, activation=None)

	def call(self, inputs, **kwargs):
		sparse_inputs = inputs
		# embedding
		sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)  # (batch_size, embed_dim * fields)
		# wide
		sparse_inputs = sparse_inputs + tf.convert_to_tensor(self.index_mapping)
		wide_inputs = {'sparse_inputs': sparse_inputs,
					   'embed_inputs': tf.reshape(sparse_embed, shape=(-1, sparse_inputs.shape[1], self.embed_dim))}
		wide_outputs = self.fm(wide_inputs)  # (batch_size, 1)
		# deep
		deep_outputs = self.dnn(sparse_embed)
		deep_outputs = self.dense(deep_outputs)  # (batch_size, 1)
		# outputs
		outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
		return outputs

	def summary(self):
		sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
		Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
