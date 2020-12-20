"""
Created on Dec 20, 2020

dnn modules

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout


class DNN(Layer):
	"""
	Deep part
	"""
	def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
		"""
		DNN part
		:param hidden_units: A list. List of hidden layer units's numbers
		:param activation: A string. Activation function
		:param dnn_dropout: A scalar. dropout number
		"""
		super(DNN, self).__init__(**kwargs)
		self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
		self.dropout = Dropout(dnn_dropout)

	def call(self, inputs, **kwargs):
		x = inputs
		for dnn in self.dnn_network:
			x = dnn(x)
		x = self.dropout(x)
		return x