"""
Created on May 18, 2021
input feature columns: sparseFeature, denseFeature
@author: Ziyao Geng(zggzy1996@163.com)
"""


def sparseFeature(feat_name, feat_num, embed_dim=4):
    """Create dictionary for sparse feature.
    Args:
        :param feat_name: A string. feature name.
        :param feat_num: A scalar(int). The total number of sparse features that do not repeat.
        :param embed_dim: A scalar(int). embedding dimension for this feature.
    :return:
    """
    return {'feat_name': feat_name, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat_name):
    """Create dictionary for dense feature.
    Args:
        :param feat_name: A string. dense feature name.
    :return:
    """
    return {'feat_name': feat_name}
