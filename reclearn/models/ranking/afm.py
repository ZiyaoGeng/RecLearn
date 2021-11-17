"""
Created on August 3, 2020
Updated on Nov 13, 2021
Reference: "Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks", IJCAI, 2017
@author: Ziyao Geng(zggzy1996@163.com)
"""
import itertools
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Dropout, Input


class AFM(Model):
    def __init__(self, feature_columns, mode, att_dim=8, activation='relu', dnn_dropout=0., embed_reg=0.):
        """Attentional Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param mode: A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
            :param att_dim: A scalar. attention vector dimension.
            :param activation: A string. Activation function of attention.
            :param dnn_dropout: A scalar. Dropout of MLP.
            :param embed_reg: A scalar. The regularization coefficient of embedding.
        :return:
        """
        super(AFM, self).__init__()
        self.feature_columns = feature_columns
        self.mode = mode
        self.embed_layers = {
            feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for feat in self.feature_columns
        }
        self.embed_dim = self.feature_columns[0]['embed_dim']
        self.field_num = len(self.feature_columns)
        if self.mode == 'att':
            self.attention_W = Dense(units=att_dim, activation=activation)
            self.attention_dense = Dense(units=1, activation=None)
        self.dropout = Dropout(dnn_dropout)
        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        # Embedding Layer
        sparse_embed = tf.concat([self.embed_layers[feat_name](value) for feat_name, value in inputs.items()], axis=-1)
        sparse_embed = tf.reshape(sparse_embed, [-1, self.field_num, self.embed_dim])  # (None, filed_num, embed_dim)
        # Pair-wise Interaction Layer
        row = []
        col = []
        for r, c in itertools.combinations(range(self.field_num), 2):
            row.append(r)
            col.append(c)
        p = tf.gather(sparse_embed, row, axis=1)  # (None, (field_num * (field_num - 1)) / 2, k)
        q = tf.gather(sparse_embed, col, axis=1)  # (None, (field_num * (field_num - 1)) / 2, k)
        bi_interaction = p * q  # (None, (field_num * (field_num - 1)) / 2, k)
        # mode
        if self.mode == 'max':
            # MaxPooling Layer
            x = tf.reduce_sum(bi_interaction, axis=1)   # (None, k)
        elif self.mode == 'avg':
            # AvgPooling Layer
            x = tf.reduce_mean(bi_interaction, axis=1)  # (None, k)
        else:
            # Attention Layer
            x = self._attention(bi_interaction)  # (None, k)
        # Output Layer
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def _attention(self, bi_interaction):
        print(bi_interaction)
        a = self.attention_W(bi_interaction)  # (None, (field_num * (field_num - 1)) / 2, embed_dim)
        a = self.attention_dense(a)  # (None, (field_num * (field_num - 1)) / 2, 1)
        a_score = tf.nn.softmax(a, axis=1)  # (None, (field_num * (field_num - 1)) / 2, 1)
        outputs = tf.reduce_sum(bi_interaction * a_score, axis=1)  # (None, embed_dim)
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()
