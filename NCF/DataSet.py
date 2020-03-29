"""
Created on Nov 22, 2019

@author: GengZiyao (zggzy1996@gmail.com)
"""

import scipy.sparse as sp
import numpy as np


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_file_as_matrix(filename):
    """
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    """
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if rating > 0:
                mat[user, item] = 1.0
            line = f.readline()
    return mat


class Dataset(object):

    def __init__(self, path):
        """
        Constructor
        """
        self.trainMatrix = load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape
