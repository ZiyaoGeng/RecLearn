"""
Created on Sept 10, 2020

modules of SASRec: attention mechanism, multi-head attention, self-attention block

@author: Ziyao Geng
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Conv1D


# def positional_encoding(seq_inputs, embed_dim):
#     encoded_vec = [pos / np.power(10000.0, 2 * i / embed_dim)
#                    for pos in range(seq_inputs.shape[-1]) for i in range(embed_dim)]
#     encoded_vec[::2] = np.sin(encoded_vec[::2])
#     encoded_vec[1::2] = np.cos(encoded_vec[1::2])
#     encoded_vec = tf.reshape(tf.convert_to_tensor(encoded_vec, dtype=tf.float32), shape=(-1, embed_dim))
#     return encoded_vec


def scaled_dot_product_attention(q, k, v, causality=True):
    """
    Attention Mechanism
    :param q: A 3d tensor with shape of (None, seq_len, depth), depth = d_model // num_heads
    :param k: A 3d tensor with shape of (None, seq_len, depth)
    :param v: A 3d tensor with shape of (None, seq_len, depth)
    :param causality: Boolean. If True, using causality, default True
    :return:
    """
    mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
    dk = tf.cast(k.shape[-1], dtype=tf.float32)
    # Scaled
    scaled_att_logits = mat_qk / tf.sqrt(dk)
    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(k, axis=-1)))  # (None, seq_len)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, q.shape[1], 1])  # (None, seq_len, seq_len)

    paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)

    # Causality
    if causality:
        diag_vals = tf.ones_like(outputs)  # (None, seq_len, seq_len)
        masks = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (None, seq_len, seq_len)

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (None, seq_len, seq_len)

    # softmax
    outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len, seq_len)

    # Query Masking
    query_masks = tf.sign(tf.abs(tf.reduce_sum(q, axis=-1)))  # (None, seq_len)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, q.shape[1]])  # (None, seq_len, seq_len)
    outputs *= query_masks

    outputs = tf.matmul(outputs, v)  # (None, seq_len, depth)

    return outputs


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, causality=True):
        """
        Multi Head Attention Mechanism
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        :param causality: Boolean. If True, using causality, default True

        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causality = causality

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, q, k, v):
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)

        # split d_model into num_heads * depth, and concatenate
        q = tf.reshape(tf.concat([tf.split(q, self.num_heads, axis=2)], axis=0),
                       (-1, q.shape[1], q.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)
        k = tf.reshape(tf.concat([tf.split(k, self.num_heads, axis=2)], axis=0),
                       (-1, k.shape[1], k.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)
        v = tf.reshape(tf.concat([tf.split(v, self.num_heads, axis=2)], axis=0),
                       (-1, v.shape[1], v.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)

        # attention
        scaled_attention = scaled_dot_product_attention(q, k, v, self.causality)  # (None * num_heads, seq_len, d_model // num_heads)

        # Reshape
        outputs = tf.concat(tf.split(scaled_attention, self.num_heads, axis=0), axis=2)  # (N, seq_len, d_model)

        return outputs


class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """
        Feed Forward Network
        :param hidden_unit: A scalar. W1
        :param d_model: A scalar. W2
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output


class SelfAttentionBlock(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., norm_training=True, causality=True):
        """
        Self Attention Block
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        """
        super(SelfAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, causality)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=1e-6, trainable=norm_training)
        self.layernorm2 = LayerNormalization(epsilon=1e-6, trainable=norm_training)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x = inputs
        # self-attention
        att_out = self.mha(x, x, x)  # （None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2