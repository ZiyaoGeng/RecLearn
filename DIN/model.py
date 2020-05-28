"""
Created on May 23, 2020

model: Deep interest network for click-through rate prediction

@author: Ziyao Geng
"""
import tensorflow as tf
import numpy as np

import os

from dice import Dice
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DIN(tf.keras.Model):
    def __init__(self, user_num, item_num, cate_num, cate_list, hidden_units):
        """
        :param user_num: 用户数量
        :param item_num: 物品数量
        :param cate_num: 物品种类数量
        :param cate_list: 物品种类列表
        :param hidden_units: 隐藏层单元
        """
        super(DIN, self).__init__()
        self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int32)
        self.hidden_units = hidden_units
        # self.user_embed = tf.keras.layers.Embedding(
        #     input_dim=user_num, output_dim=hidden_units, embeddings_initializer='random_uniform',
        #     embeddings_regularizer=tf.keras.regularizers.l2(0.01), name='user_embed')
        self.item_embed = tf.keras.layers.Embedding(
            input_dim=item_num, output_dim=self.hidden_units, embeddings_initializer='random_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(0.01), name='item_embed')
        self.cate_embed = tf.keras.layers.Embedding(
            input_dim=cate_num, output_dim=self.hidden_units, embeddings_initializer='random_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(0.01), name='cate_embed'
        )
        self.dense = tf.keras.layers.Dense(self.hidden_units)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.att_dense1 = tf.keras.layers.Dense(80, activation='sigmoid')
        self.att_dense2 = tf.keras.layers.Dense(40, activation='sigmoid')
        self.att_dense3 = tf.keras.layers.Dense(1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(80, activation='sigmoid')
        self.activation1 = tf.keras.layers.PReLU()
        # self.activation1 = Dice()
        self.dense2 = tf.keras.layers.Dense(40, activation='sigmoid')
        self.activation2 = tf.keras.layers.PReLU()
        # self.activation2 = Dice()
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        user, item, hist, sl = inputs[0], tf.squeeze(inputs[1], axis=1), inputs[2], tf.squeeze(inputs[3], axis=1)
        # user_embed = self.u_embed(user)
        item_embed = self.concat_embed(item)
        hist_embed = self.concat_embed(hist)
        # 经过attention的物品embedding
        hist_att_embed = self.attention(item_embed, hist_embed, sl)
        hist_att_embed = self.bn1(hist_att_embed)
        hist_att_embed = tf.reshape(hist_att_embed, [-1, self.hidden_units * 2])
        u_embed = self.dense(hist_att_embed)
        item_embed = tf.reshape(item_embed, [-1, item_embed.shape[-1]])
        # 联合用户行为embedding、候选物品embedding、【用户属性、上下文内容特征】
        embed = self.concat2([u_embed, item_embed])
        x = self.bn2(embed)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dense3(x)
        outputs = tf.nn.sigmoid(x)
        return outputs

    def summary(self):
        user = tf.keras.Input(shape=(1,), dtype=tf.int32)
        item = tf.keras.Input(shape=(1,), dtype=tf.int32)
        sl = tf.keras.Input(shape=(1,), dtype=tf.int32)
        hist = tf.keras.Input(shape=(431,), dtype=tf.int32)
        tf.keras.Model(inputs=[user, item, hist, sl], outputs=self.call([user, item, hist, sl])).summary()

    def concat_embed(self, item):
        """
        拼接物品embedding和物品种类embedding
        :param item: 物品id
        :return: 拼接后的embedding
        """
        # cate = tf.transpose(tf.gather_nd(self.cate_list, [item]))
        cate = tf.gather(self.cate_list, item)
        cate = tf.squeeze(cate, axis=1) if cate.shape[-1] == 1 else cate
        item_embed = self.item_embed(item)
        item_cate_embed = self.cate_embed(cate)
        embed = self.concat([item_embed, item_cate_embed])
        return embed

    def attention(self, queries, keys, keys_length):
        """
        activation unit
        :param queries: item embedding
        :param keys: hist embedding
        :param keys_length: the number of hist_embed
        :return:
        """
        # 候选物品的隐藏向量维度，hidden_unit * 2
        queries_hidden_units = queries.shape[-1]
        # 每个历史记录的物品embed都需要与候选物品的embed拼接，故候选物品embed重复keys.shape[1]次
        # keys.shape[1]为最大的序列长度，即431，为了方便矩阵计算
        # [None, 431 * hidden_unit * 2]
        queries = tf.tile(queries, [1, keys.shape[1]])
        # 重塑候选物品embed的shape
        # [None, 431, hidden_unit * 2]
        queries = tf.reshape(queries, [-1, keys.shape[1], queries_hidden_units])
        # 拼接候选物品embed与hist物品embed
        # [None, 431, hidden * 2 * 4]
        embed = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        # 全连接, 得到权重W
        d_layer_1 = self.att_dense1(embed)
        d_layer_2 = self.att_dense2(d_layer_1)
        # [None, 431, 1]
        d_layer_3 = self.att_dense3(d_layer_2)
        # 重塑输出权重类型, 每个hist物品embed有对应权重值
        # [None, 1, 431]
        outputs = tf.reshape(d_layer_3, [-1, 1, keys.shape[1]])

        # Mask
        # 此处将为历史记录的物品embed令为True
        # [None, 431]
        key_masks = tf.sequence_mask(keys_length, keys.shape[1])
        # 增添维度
        # [None, 1, 431]
        key_masks = tf.expand_dims(key_masks, 1)
        # 填充矩阵
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # 构造输出矩阵，其实就是为了实现【sum pooling】。True即为原outputs的值，False为上述填充值，为很小的值，softmax后接近0
        # [None, 1, 431] ----> 每个历史浏览物品的权重
        outputs = tf.where(key_masks, outputs, paddings)
        # Scale，keys.shape[-1]为hist_embed的隐藏单元数
        outputs = outputs / (keys.shape[-1] ** 0.5)
        # Activation，归一化
        outputs = tf.nn.softmax(outputs)
        # 对hist_embed进行加权
        # [None, 1, 431] * [None, 431, hidden_unit * 2] = [None, 1, hidden_unit * 2]
        outputs = tf.matmul(outputs, keys)
        return outputs