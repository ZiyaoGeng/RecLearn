"""
Created on Nov 14, 2021
Loss function.
@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf


def get_loss(pos_scores, neg_scores, loss_name, gamma=None):
    """Get loss scores.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param loss_name: A string such as 'bpr_loss', 'hing_loss' and etc.
        :param gamma: A scalar(int). If loss_name == 'hinge_loss', the gamma must be valid.
    :return:
    """
    pos_scores = tf.tile(pos_scores, [1, neg_scores.shape[1]])
    if loss_name == 'bpr_loss':
        loss = bpr_loss(pos_scores, neg_scores)
    elif loss_name == 'hinge_loss':
        loss = hinge_loss(pos_scores, neg_scores, gamma)
    else:
        loss = binary_cross_entropy_loss(pos_scores, neg_scores)
    return loss


def bpr_loss(pos_scores, neg_scores):
    """bpr loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, 1].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
    return loss


def hinge_loss(pos_scores, neg_scores, gamma=0.5):
    """hinge loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
        :param gamma: A scalar(int).
    :return:
    """
    loss = tf.reduce_mean(tf.nn.relu(neg_scores - pos_scores + gamma))
    return loss


def binary_cross_entropy_loss(pos_scores, neg_scores):
    """binary cross entropy loss.
    Args:
        :param pos_scores: A tensor with shape of [batch_size, neg_num].
        :param neg_scores: A tensor with shape of [batch_size, neg_num].
    :return:
    """
    loss = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) - tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
    return loss