import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
    Attention Mechanism
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
    """
    x = tf.reshape(x, (-1, seq_len, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])