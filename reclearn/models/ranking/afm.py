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
    def __init__(self, fea_cols, mode, att_vector=8, activation='relu', dropout=0.5, embed_reg=1e-6):
        """
        AFM
        :param fea_cols: A list. sparse column feature information.
        :param mode: A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param att_vector: A scalar. attention vector.
        :param activation: A string. Activation function of attention.
        :param dropout: A scalar. Dropout.
        :param embed_reg: A scalar. the regularizer of embedding
        """
        super(AFM, self).__init__()
        self.fea_cols = fea_cols
        self.mode = mode
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.fea_cols)
        }
        if self.mode == 'att':
            self.attention_W = Dense(units=att_vector, activation=activation, use_bias=True)
            self.attention_dense = Dense(units=1, activation=None)
        self.dropout = Dropout(dropout)
        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        # Input Layer
        sparse_inputs = inputs
        # Embedding Layer
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), perm=[1, 0, 2])  # (None, len(sparse_inputs), embed_dim)
        # Pair-wise Interaction Layer
        row = []
        col = []
        for r, c in itertools.combinations(range(len(self.sparse_feature_columns)), 2):
            row.append(r)
            col.append(c)
        p = tf.gather(embed, row, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        q = tf.gather(embed, col, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        bi_interaction = p * q  # (None, (len(sparse) * len(sparse) - 1) / 2, k)
        # mode
        if self.mode == 'max':
            # MaxPooling Layer
            x = tf.reduce_sum(bi_interaction, axis=1)   # (None, k)
        elif self.mode == 'avg':
            # AvgPooling Layer
            x = tf.reduce_mean(bi_interaction, axis=1)  # (None, k)
        else:
            # Attention Layer
            x = self.attention(bi_interaction)  # (None, k)
        # Output Layer
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.fea_cols),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

    def attention(self, bi_interaction):
        a = self.attention_W(bi_interaction)  # (None, (len(sparse) * len(sparse) - 1) / 2, t)
        a = self.attention_dense(a)  # (None, (len(sparse) * len(sparse) - 1) / 2, 1)
        a_score = tf.nn.softmax(a, axis=1)  # (None, (len(sparse) * len(sparse) - 1) / 2, 1)
        outputs = tf.reduce_sum(bi_interaction * a_score, axis=1)  # (None, embed_dim)
        return outputs
