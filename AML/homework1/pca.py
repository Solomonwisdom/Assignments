# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
from knn import KNN
from utils import load_dataset


def pca(data_mat, n_components):

    # 1.对所有样本进行中心化（所有样本属性减去属性的平均值）
    mean_vals = np.mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals

    # 2.计算样本的协方差矩阵 XXT
    covmat = np.cov(mean_removed, rowvar=False)
    # print(covmat)

    # 3.对协方差矩阵做特征值分解，求得其特征值和特征向量，并将特征值从大到小排序，筛选出前n_components个
    eig_vals, eig_vects = np.linalg.eig(np.mat(covmat))
    eig_vals = np.argsort(eig_vals)
    eig_vals = eig_vals[:-(n_components+1):-1]    # 取前topN大的特征值的索引
    red_eig_vects = eig_vects[:, eig_vals]        # 取前topN大的特征值所对应的特征向量

    return mean_vals, red_eig_vects


def transform(eig_vectors, mean_vals, data_mat):
    mean_removed = data_mat - mean_vals
    return np.matmul(mean_removed, eig_vectors)

# ---------------------------- main ---------------------------- #


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
    mean_vals, eig_vectors = pca(train_X, n_components)
    train_X_low = transform(eig_vectors, mean_vals, train_X)
    test_X_low = transform(eig_vectors, mean_vals, test_X)
    one_NN = KNN(train_X_low, train_Y, 1, n_components)
    test_y_pred = one_NN.predict(test_X_low)
    print('pca(k={0:d},data={1:s}): {2:.4f}%'.
          format(n_components, fileNamePrefix, one_NN.score(test_Y, test_y_pred)*100))
