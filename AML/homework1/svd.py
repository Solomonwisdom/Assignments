# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import os
import sys
from knn import KNN
from utils import load_dataset


def SVD(data_mat, k):
    # singular value decomposition
    U, s, V = la.svd(data_mat)
    # choose top k important singular values (or eigens)
    return V[0:k, :].T


# -------------------- main --------------------- #


DIR = './two datasets/'
if __name__ == "__main__":
    if len(sys.argv) < 2:
        fileNamePrefix = 'sonar'
    else:
        fileNamePrefix = sys.argv[1]
    if len(sys.argv) < 3:
        n_components = 10
    else:
        n_components = int(sys.argv[2])
    train_X, train_Y = load_dataset(os.path.join(DIR, fileNamePrefix+"-train.txt"))
    test_X, test_Y = load_dataset(os.path.join(DIR, fileNamePrefix + "-test.txt"))
    VT = SVD(train_X, n_components)
    train_X_low = np.matmul(train_X, VT)
    test_X_low = np.matmul(test_X, VT)
    one_NN = KNN(train_X_low, train_Y, 1, n_components)
    test_y_pred = one_NN.predict(test_X_low)
    print('svd(k={0:d},data={1:s}): {2:.4f}%'.
          format(n_components, fileNamePrefix, one_NN.score(test_Y, test_y_pred) * 100))
