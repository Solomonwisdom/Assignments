# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.sparse
import sys

from time import time
from knn import KNN
from utils import load_dataset, PeekablePriorityQueue, get_distance


def mds(data, target):
    # mds 算法的具体实现
    # data：需要降维的矩阵
    # target：目标维度
    # return：降维后的矩阵
    data_count = len(data)
    if target > data_count:
        target = data_count
    val_dist_i_j = 0.0
    vec_dist_i_2 = np.zeros([data_count], np.float64)
    vec_dist_j_2 = np.zeros([data_count], np.float64)
    mat_b = np.zeros([data_count, data_count], np.float64)
    mat_distance = get_distance(data)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            dist_ij_2 = np.square(mat_distance[idx][sub_idx])
            val_dist_i_j += dist_ij_2
            vec_dist_i_2[idx] += dist_ij_2
            vec_dist_j_2[sub_idx] += dist_ij_2 / data_count
        vec_dist_i_2[idx] /= data_count
    val_dist_i_j /= np.square(data_count)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            dist_ij_2 = np.square(mat_distance[idx][sub_idx])
            mat_b[idx][sub_idx] = -0.5 * (dist_ij_2 - vec_dist_i_2[idx] - vec_dist_j_2[sub_idx] + val_dist_i_j)
    a, v = np.linalg.eig(mat_b)
    list_idx = np.argpartition(a, data_count - target)[-target:]
    a = np.diag(np.maximum(a[list_idx], 0.0))
    return np.matmul(v[:, list_idx], np.sqrt(a))


def isomap(data, target, k):
    # isomap 算法的具体实现
    # data：需要降维的矩阵
    # target：目标维度
    # k：k 近邻算法中的超参数
    # return：降维后的矩阵
    inf = float('inf')
    data_count = data.shape[0]
    if k >= data_count:
        raise ValueError('K的值最大为数据个数 - 1')
    distance_mat = get_distance(data)
    knn_map = np.ones([data_count, data_count], np.float64) * inf
    adjlist = [[] for _ in range(data_count)]
    for idx in range(data_count):
        top_k = np.argpartition(distance_mat[idx], k)[:k + 1]
        # 使用无向边
        knn_map[idx][top_k] = distance_mat[idx][top_k]
        for p in top_k:
            p = int(p)
            knn_map[p][idx] = distance_mat[p][idx]
            if p != idx:
                adjlist[p].append(idx)
                adjlist[idx].append(p)
    for idx in range(data_count):
        adjlist[idx] = [(dst, distance_mat[idx][dst]) for dst in np.unique(adjlist[idx])]
    if not is_connect(adjlist, 0, data_count):
        return None
    knn_map = scipy.sparse.csgraph.shortest_path(knn_map, directed=False)
    return mds(knn_map, target)


def is_connect(adjlist, src, num):
    pqueue = PeekablePriorityQueue()
    pqueue.put((0, src))
    done = np.array([False]*num)
    dist = np.ones((num,), np.float64)*float('inf')
    while not pqueue.empty():
        sdist, u = pqueue.get()
        if done[u]:
            continue
        done[u] = True
        for dst, tdist in adjlist[u]:
            if dist[dst] > sdist + tdist:
                dist[dst] = sdist + tdist
                pqueue.put((dist[dst], dst))
    if not done.all():
        print("not complete", done.shape[0]-np.count_nonzero(done))
        return False
    return True


# ---------------------------- main ---------------------------- #


DIR = './two datasets/'
if __name__ == "__main__":
    start = time()
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
    data = np.concatenate((train_X, test_X))
    k = 4
    while True:
        data_low = isomap(data, n_components, k)
        # train_X_low = isomap(train_X, n_components, k)
        # test_X_low = isomap(test_X, n_components, k)
        # if train_X_low is not None and test_X_low is not None:
        if data_low is not None:
            break
        k += 1
    print('k of the knn:', k)
    train_X_low, test_X_low = data_low[:train_X.shape[0], :], data_low[train_X.shape[0]:, :]
    one_NN = KNN(train_X_low, train_Y, 1, n_components)
    test_y_pred = one_NN.predict(test_X_low)
    print('isomap(k={0:d},data={1:s}): {2:.4f}%'.
          format(n_components, fileNamePrefix, one_NN.score(test_Y, test_y_pred)*100))
    stop = time()
    print('time spent:', str(stop - start) + "s")

