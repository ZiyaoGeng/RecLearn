"""
Created on July 27, 2020
Updated on Nov 14, 2021
Reference: "Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features", KDD, 2016
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import Residual_Units


class Deep_Crossing(Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., embed_reg=0.):
        """Deep&Crossing.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param hidden_units: A list. A list of MLP hidden units.
            :param dnn_dropout: A scalar. Dropout of resnet.
            :param embed_reg: A scalar. The regularization coefficient of embedding.
        :return:
        """
        super(Deep_Crossing, self).__init__()
        self.feature_columns = feature_columns
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        # the total length of embedding layers
        embed_layers_len = sum([feat['embed_dim'] for feat in self.feature_columns])
        self.res_network = [Residual_Units(unit, embed_layers_len) for unit in hidden_units]
        self.res_dropout = Dropout(dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        r = sparse_embed
        for res in self.res_network:
            r = res(r)
        r = self.res_dropout(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()