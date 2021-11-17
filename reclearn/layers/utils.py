import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """Attention Mechanism Function.
    Args:
        :param q: A 3d/4d tensor with shape of (None, ..., seq_len, dim)
        :param k: A 3d/4d tensor with shape of (None, ..., seq_len, dim)
        :param v: A 3d/4d tensor with shape of (None, ..., seq_len, dim)
        :param mask: A 3d/4d tensor with shape of (None, ..., seq_len, 1)
    :return:
    """
    mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
    # Scaled
    dk = tf.cast(k.shape[-1], dtype=tf.float32)
    scaled_att_logits = mat_qk / tf.sqrt(dk)

    paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)  # (None, seq_len, seq_len)
    outputs = tf.where(tf.equal(mask, tf.zeros_like(mask)), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
    # softmax
    outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len, seq_len)
    outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)

    return outputs


def split_heads(x, seq_len, num_heads, depth):
    """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    Args:
        :param x: A Tensor with shape of [batch_size, seq_len, num_heads * depth]
        :param seq_len: A scalar(int).
        :param num_heads: A scalar(int).
        :param depth: A scalar(int).
    :return: A tensor with shape of [batch_size, num_heads, seq_len, depth]
    """
    x = tf.reshape(x, (-1, seq_len, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])


def index_mapping(inputs_dict, map_dict):
    """Feature index mapping
    Args:
        :param inputs_dict: A dict such as {'I1': [], 'I2': [], ...}
        :param map_dict: A dict such as {'I1': 0, 'I2': 100, ...}
    :return: new inputs tensor.
    """
    outputs_dict = {}
    for key, value in inputs_dict.items():
        if map_dict.get(key) is None:
            raise ValueError("map dict error!")
        outputs_dict[key] = tf.reshape(value + tf.convert_to_tensor(map_dict[key]), [-1, 1])
    return outputs_dict