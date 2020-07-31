"""
Created on July 20, 2020

model: Product-based Neural Networks for User Response Prediction

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, optimizers, losses, regularizers
from tensorflow.keras.layers import Embedding, Dropout, Flatten, Dense, Input

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FM(layers.Layer):
	"""
	Wide part
	"""
	def __init__(self, k):
		"""

		:param k: the dimension of the latent vector
		"""
		super(FM, self).__init__()
		self.k = k

	def build(self, input_shape):
		self.w0 = self.add_weight(name='w0', shape=(1,), initializer=tf.zeros_initializer())
		self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
								 regularizer=regularizers.l2(0.01))
		self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
								 regularizer=regularizers.l2(0.01))

	def call(self, inputs):
		# first order
		first_order = self.w0 + tf.matmul(inputs, self.w)
		# second order
		second_order = 0.5 * tf.reduce_sum(
			tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
			tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)
		return first_order + second_order


class MLP(layers.Layer):
	"""
	Deep part
	"""
	def __init__(self, hidden_units, dropout_deep):
		"""

		:param hidden_units: list of hidden layer units's numbers
		:param dropout_deep: dropout number
		"""
		super(MLP, self).__init__()
		self.dnn_network = [Dense(units=unit, activation='relu') for unit in hidden_units]
		self.dropout = Dropout(dropout_deep)

	def call(self, inputs):
		x = inputs
		for dnn in self.dnn_network:
			x = dnn(x)
		x = self.dropout(x)
		return x


class DeepFM(keras.Model):
	def __init__(self, feature_columns, k, hidden_units=None, dropout_deep=0.5):
		super(DeepFM, self).__init__()
		if hidden_units is None:
			hidden_units = [200, 200, 200]
		self.dense_feature_columns, self.sparse_feature_columns = feature_columns
		# embedding layer
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
										 output_dim=feat['embed_dim'], embeddings_initializer='random_uniform')
			for i, feat in enumerate(self.sparse_feature_columns)
		}
		self.fm = FM(k)
		self.mlp = MLP(hidden_units, dropout_deep)
		self.dense = Dense(1, activation=None)
		self.w1 = self.add_weight(name='wide_weight', shape=(1,))
		self.w2 = self.add_weight(name='deep_weight', shape=(1,))

	def call(self, inputs):
		dense_inputs, sparse_inputs = inputs
		stack = dense_inputs
		for i in range(sparse_inputs.shape[1]):
			embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
			stack = tf.concat([stack, embed_i], axis=-1)
		wide_outputs = self.fm(stack)
		deep_outputs = self.mlp(stack)
		deep_outputs = self.dense(deep_outputs)
		outputs = tf.nn.sigmoid(tf.add(self.w1 * wide_outputs, self.w2 * deep_outputs))
		return outputs

	def summary(self):
		dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
		sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
		keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


def main():
	dense = [{'name': 'c1'}, {'name': 'c2'}, {'name': 'c3'}, {'name': 'c4'}]
	sparse = [{'feat_num': 100, 'embed_dim': 256}, {'feat_num': 200, 'embed_dim': 256}]
	columns = [dense, sparse]
	model = DeepFM(columns, 10, [200, 200, 200], 0.5)
	model.summary()


# main()
