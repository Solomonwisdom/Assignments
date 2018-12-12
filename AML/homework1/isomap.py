# -*- coding: utf-8 -*-

import numpy as np
import os
import sys

from time import time
from knn import KNN
from utils import load_dataset, PeekablePriorityQueue, get_distance


def mds(dist, target):
    # [vec, val] = eigs(-.5*(D.^2 - sum(D.^2)'*ones(1,N)/N -
    # ones(N,1)*sum(D.^2)/N + sum(sum(D.^2))/(N^2)), max(dims), 'LR', opt);
    # mds 算法的具体实现
    # data：需要降维的矩阵
    # target：目标维度
    # return：降维后的矩阵
    dim = dist.shape[0]
    if target > dim:
        target = dim
    dist_ij = np.asarray(dist, np.float64)
    dist_ij_2 = dist_ij ** 2
    dist_i_2 = np.dot(np.mean(dist_ij_2, axis=1).reshape((dim, 1)), np.ones((1, dim)))
    dist_j_2 = np.dot(np.ones((dim, 1)), np.mean(dist_ij_2, axis=0).reshape((1, dim)))
    dist_2 = np.mean(np.mean(dist_ij_2, axis=1))
    b = -0.5*(dist_ij_2-dist_i_2-dist_j_2+dist_2)
    eig_val, eig_vec = np.linalg.eig(b)
    list_idx = np.argsort(eig_val)[-target:]
    return np.dot(eig_vec[:, list_idx], np.sqrt(np.diag(eig_val[list_idx])))


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
        for p in top_k:
            p = int(p)
            if p != idx:
                adjlist[p].append(idx)
                adjlist[idx].append(p)
    for idx in range(data_count):
        adjlist[idx] = [(dst, distance_mat[idx][dst]) for dst in np.unique(adjlist[idx])]
    for idx in range(data_count):
        if not dijkstra(knn_map, adjlist, idx):
            return None
    return mds(knn_map, target)


def dijkstra(dist, adjlist, src):
    pqueue = PeekablePriorityQueue()
    pqueue.put((0, src))
    dist[src][src] = 0
    done = np.array([False]*dist.shape[0])
    while not pqueue.empty():
        sdist, u = pqueue.get()
        if done[u]:
            continue
        done[u] = True
        for dst, tdist in adjlist[u]:
            if dist[src][dst] > sdist + tdist:
                dist[src][dst] = sdist + tdist
                pqueue.put((dist[src][dst], dst))
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
    k = 6 if fileNamePrefix == 'sonar' else 4
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

