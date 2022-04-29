"""
Created on July 31, 2020
Updated on Nov 14, 2021
Reference: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", 2017, IJCAI
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer

from reclearn.layers import New_FM, MLP
from reclearn.layers.utils import index_mapping


class DeepFM(Model):
	def __init__(self, feature_columns, hidden_units=(200, 200, 200), activation='relu',
				 dnn_dropout=0., fm_w_reg=0., embed_reg=0.):
		"""DeepFM
		Args:
			:param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
			:param hidden_units: A list. A list of MLP hidden units.
			:param dnn_dropout: A scalar. Dropout of MLP.
			:param activation: A string. Activation function of MLP.
			:param fm_w_reg: A scalar. The regularization coefficient of w in fm.
			:param embed_reg: A scalar. The regularization coefficient of embedding.
		:return
		"""
		super(DeepFM, self).__init__()
		self.feature_columns = feature_columns
		self.embed_layers = {
			feat['feat_name']: Embedding(input_dim=feat['feat_num'],
										 input_length=1,
										 output_dim=feat['embed_dim'],
										 embeddings_initializer='random_normal',
										 embeddings_regularizer=l2(embed_reg))
			for feat in self.feature_columns
		}
		self.map_dict = {}
		self.feature_length = 0
		self.field_num = len(self.feature_columns)
		for feat in self.feature_columns:
			self.map_dict[feat['feat_name']] = self.feature_length
			self.feature_length += feat['feat_num']
		self.embed_dim = self.feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim
		self.fm = New_FM(self.feature_length, fm_w_reg)
		self.mlp = MLP(hidden_units, activation, dnn_dropout)
		self.dense = Dense(1, activation=None)

	def call(self, inputs):
		# embedding,  (batch_size, embed_dim * fields)
		sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
		# wide
		sparse_inputs = index_mapping(inputs, self.map_dict)
		wide_inputs = {'sparse_inputs': sparse_inputs,
					   'embed_inputs': tf.reshape(sparse_embed, shape=(-1, self.field_num, self.embed_dim))}
		wide_outputs = tf.reshape(self.fm(wide_inputs), [-1, 1])  # (batch_size, 1)
		# deep
		deep_outputs = self.mlp(sparse_embed)
		deep_outputs = tf.reshape(self.dense(deep_outputs), [-1, 1])  # (batch_size, 1)
		# outputs
		outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
		return outputs

	def summary(self):
		inputs = {
			feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
			for feat in self.feature_columns
		}
		Model(inputs=inputs, outputs=self.call(inputs)).summary()