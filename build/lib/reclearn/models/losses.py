import tensorflow as tf


def get_loss(pos_scores, neg_scores, loss_name, gamma=None):
    if loss_name == 'bpr_loss':
        loss = bpr_loss(pos_scores, neg_scores)
    elif loss_name == 'hinge_loss':
        loss = hinge_loss(pos_scores, neg_scores, gamma)
    else:
        loss = binary_entropy_loss(pos_scores, neg_scores)
    return loss


def bpr_loss(pos_scores, neg_scores):
    loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores)))
    return loss


def hinge_loss(pos_scores, neg_scores, gamma=0.5):
    loss = tf.reduce_mean(tf.nn.relu(neg_scores - pos_scores + gamma))
    return loss


def binary_entropy_loss(pos_scores, neg_scores):
    loss = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_scores)) - tf.math.log(1 - tf.nn.sigmoid(neg_scores))) / 2
    return loss