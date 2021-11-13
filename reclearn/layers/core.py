"""
Created on Nov 07, 2021
core layers
@author: Ziyao Geng(zggzy1996@163.com)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D
from tensorflow.keras.regularizers import l2

from reclearn.layers.utils import scaled_dot_product_attention, split_heads


class MLP(Layer):
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., is_batch_norm=False):
        """
        Multilayer Perceptron
        :param hidden_units: A list. The list of hidden layer units's numbers.
        :param activation: A string. The name of activation function, like 'relu', 'sigmoid' and so on.
        :param dnn_dropout: A scalar. The rate of dropout .
        :param is_batch_norm: A boolean. Whether using batch normalization or not.
        """
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.is_batch_norm = is_batch_norm
        self.bt = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        if self.is_batch_norm:
            x = self.bt(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        """
        Multi Head Attention Mechanism
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, q, k, v, mask):
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)
        # split d_model into num_heads * depth
        seq_len, d_model = q.shape[1], q.shape[2]
        q = split_heads(q, seq_len, self.num_heads, q.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        k = split_heads(k, seq_len, self.num_heads, k.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        v = split_heads(v, seq_len, self.num_heads, v.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        # mask
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_heads, 1, 1])  # (None, num_heads, seq_len, 1)
        # attention
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)  # (None, num_heads, seq_len, d_model // num_heads)
        # reshape
        outputs = tf.reshape(tf.transpose(scaled_attention, [0, 2, 1, 3]), [-1, seq_len, d_model])  # (None, seq_len, d_model)
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


class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., layer_norm_eps=1e-6):
        """
        Encoder Layer
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param layer_norm_eps: A scalar. Small float added to variance to avoid dividing by zero.
        """
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x, mask = inputs
        # self-attention
        att_out = self.mha(x, x, x, mask)  # (None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)  # (None, seq_len, d_model)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2


class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights',
            initializer='random_normal')

    def call(self, inputs):
        q, k, v, mask = inputs
        # pos encoding
        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        # Nonlinear transformation
        q = tf.nn.relu(tf.matmul(q, self.W))  # (None, seq_len, dim)
        k = tf.nn.relu(tf.matmul(k, self.W))  # (None, seq_len, dim)
        mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        mask = tf.tile(mask, [1, 1, q.shape[1]])  # (None, seq_len, seq_len)
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs, axis=-1)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class FM_Layer(Layer):
    def __init__(self, fea_cols, k, w_reg=1e-8, v_reg=1e-8):
        """
        Factorization Machines
        :param fea_cols: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.fea_cols = fea_cols
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.fea_cols:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        # mapping
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs