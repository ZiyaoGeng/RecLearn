"""
Created on July 9, 2020

model: Wide & Deep Learning for Recommender Systems

@author: Ziyao Geng
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.experimental import WideDeepModel
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class WideDeep(tf.keras.Model):
    """
    Wide&Deep model
    """
    def __init__(self, user_num, item_num, cate_num, cate_list, embed_unit):
        """
        :param user_num: 用户数量
        :param item_num: 物品数量
        :param cate_num: 物品种类数量
        :param cate_list: 物品种类列表
        :param embed_unit: embedding维度
        """
        super(WideDeep, self).__init__()
        self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int32)
        self.embed_unit = embed_unit
        self.item_embed = Embedding(
            input_dim=item_num, output_dim=embed_unit,
            embeddings_initializer='random_uniform', embeddings_regularizer=regularizers.l2(0.001)
        )
        self.cate_embed = Embedding(
            input_dim=cate_num, output_dim=embed_unit,
            embeddings_initializer='random_uniform', embeddings_regularizer=regularizers.l2(0.001)
        )
        self.concat1 = Concatenate(axis=-1)
        self.pooling = GlobalAveragePooling1D()
        self.bn1 = BatchNormalization()
        self.concat2 = Concatenate(axis=-1)
        self.dense_wide = Dense(units=1)
        self.concat3 = Concatenate(axis=-1)
        self.bn2 = BatchNormalization()
        self.dense_deep1 = Dense(units=80, activation='relu')
        self.dense_deep2 = Dense(units=40, activation='relu')
        self.dense_deep3 = Dense(units=1)

    def call(self, inputs):
        # user:id, item:id, hist:user history, sl:effective user history length
        # item, sl 输入的时候是一个向量【batch中为矩阵】，故需要降维为标量【batch中为向量】
        user, item, hist, sl = inputs[0], tf.squeeze(inputs[1], axis=1), inputs[2], tf.squeeze(inputs[3], axis=1)
        # 物品对应种类
        item_cate = tf.gather(self.cate_list, item)
        # 物品embedding, shape=[None, embed_unit * 2]
        item_embed = self.concat1([self.item_embed(item), self.cate_embed(item_cate)])
        # 历史行为的种类
        hist_cate = tf.gather(self.cate_list, hist)
        # 历史行为embedding, shape=[None, len(hist), embed_unit * 2]
        hist_embed = self.concat1([self.item_embed(hist), self.cate_embed(hist_cate)])

        # mask
        # 构造掩码向量，shape=[None, len(hist)]，且前sl个为1
        mask = tf.sequence_mask(sl, hist_embed.shape[1], dtype=tf.float32)
        # 扩展维度，shape=[None, len(hist), 1]
        mask = tf.expand_dims(mask, -1)
        # 多次复制，shape=[None, len(hist), embed_unit * 2]
        mask = tf.tile(mask, [1, 1, hist_embed.shape[2]])
        # 转化为真正的hist_embed
        hist_embed *= mask
        # 池化
        # hist_embed = tf.reduce_sum(hist_embed, 1)
        hist_embed = self.pooling(hist_embed)
        hist_embed = self.bn1(hist_embed)
        user_embed = hist_embed
        # 拼接所有的embed
        embed = self.concat2([user_embed, item_embed])

        # Wide
        # 特征工程
        wide = self.concat3([tf.gather(user_embed, [0], axis=-1) * tf.gather(item_embed, [0], axis=-1),
                          tf.gather(user_embed, [self.embed_unit*2-1], axis=-1) * tf.gather(item_embed, [self.embed_unit*2-1], axis=-1),
                          tf.gather(user_embed, [self.embed_unit], axis=-1) *
                          tf.gather(item_embed, [self.embed_unit], axis=-1)])
        wide_out = self.dense_wide(wide)
        # Deep
        deep = self.bn2(embed)
        deep = self.dense_deep1(deep)
        deep = self.dense_deep2(deep)
        deep_out = self.dense_deep3(deep)
        outputs = tf.nn.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

    def summary(self):
        user = tf.keras.Input(shape=(1,), dtype=tf.int32)
        item = tf.keras.Input(shape=(1,), dtype=tf.int32)
        sl = tf.keras.Input(shape=(1,), dtype=tf.int32)
        # 431为用户行为的最长长度
        hist = tf.keras.Input(shape=(431,), dtype=tf.int32)
        tf.keras.Model(inputs=[user, item, hist, sl], outputs=self.call([user, item, hist, sl])).summary()


def main():
    cate_list = np.arange(100)
    model = WideDeep(100, 100, 5, cate_list, 6)
    model.summary()








