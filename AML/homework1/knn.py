# -*- coding: utf-8 -*-

import numpy as np
import operator
from utils import PeekablePriorityQueue


def _square_distance(v1, v2):
    return np.sum(np.square(v1 - v2))


class KNN(object):

    def __init__(self, x, y, k=1, dim=10):
        self.k = k
        self.y = y
        self.x = x
        # if k > 1:
        #     self.kdtree = KDTree(x, y, dim)
        #     self.kdtree.build(0, len(y)-1, 0, 0)

    def _vote(self, ys):
        vote_dict = {}
        # 对k近邻点的类别计数，取数量最多的类别为预测值
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_dict[0][0]

    def knn(self, vec):
        dist_arr = np.sum(np.square(np.tile(vec, (self.x.shape[0], 1)) - self.x), axis=1)
        dist_arr = np.array(dist_arr).reshape((dist_arr.shape[0],))
        return np.argpartition(dist_arr, self.k-1)[:self.k]

    def predict(self, x):
        y_pred = []
        for elem in x:
            top_k_index = self.knn(elem)
            y_pred.append(self._vote(self.y[top_k_index]))
            # if self.k == 1:
            #     dist_arr = np.sum(np.square(np.tile(elem, (self.x.shape[0], 1)) - self.x), axis=1)
            #     y_pred.append(self.y[np.argmin(dist_arr)])
            # else:
            #     self.kdtree.query(np.array(elem).reshape((x.shape[1],)), self.k, 0, 0)
            #     top_k_index = []
            #     while self.kdtree.pqueue.empty() is False:
            #         top_k_index.append(self.kdtree.pqueue.get()[1])
            #     y_pred.append(self._vote(self.y[top_k_index]))
        return np.array(y_pred)

    def score(self, y_true=None, y_pred=None):
        if y_true is None and y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score = np.sum([true == pred for true, pred in zip(y_true, y_pred)])/len(y_true)
        return score


class Node:

    idx = 0

    def __init__(self, index, feature):
        self.index = index
        self.feature = feature

    # def __cmp__(self, other):
    #     return other.feature[Node.idx]-self.feature[Node.idx]

    def __lt__(self, other):
        return self.feature[Node.idx] < other.feature[Node.idx]


class KDTree:

    def __init__(self, x, y, k):
        self.k = k
        self.Y = y
        self.X = np.array([Node(i, np.array(elem).reshape((k,))) for i, elem in enumerate(x)])
        self.data = list()
        self.left = list()
        self.right = list()
        self.tot = 0
        self.closet = -1
        self.pqueue = PeekablePriorityQueue()

    def build(self, l, r, rt, dept):
        if l > r:
            return
        Node.idx = dept % self.k
        mid = (l+r) // 2
        self.X[l:r+1] = np.partition(self.X[l:r+1], mid-l)
        self.data.append(self.X[mid])
        self.tot += 1
        lson = self.tot
        self.left.append(-1)
        self.right.append(-1)
        self.build(l, mid-1, lson, dept+1)
        if self.tot > lson:
            self.left[rt] = lson
        rson = self.tot
        self.build(mid+1, r, rson, dept+1)
        if self.tot > rson:
            self.right[rt] = rson

    def query(self, p, m, rt, dept):
        if rt == -1:
            return
        cur = (-_square_distance(self.data[rt].feature, p), self.data[rt].index)
        dim = dept % self.k
        fg = False
        x = self.left[rt]
        y = self.right[rt]
        if p[dim] >= self.data[rt].feature[dim]:
            tmp = x
            x = y
            y = tmp
        if x != -1:
            self.query(p, m, x, dept+1)

        if m == 1:
            if self.closet == -1:
                self.closet = cur
                fg = True
            else:
                if cur[0] < self.closet[0]:
                    self.closet = cur
                if np.square(p[dim]-self.data[rt].feature[dim]) < self.closet[0]:
                    fg = True
        else:
            if self.pqueue.qsize() < m:
                self.pqueue.put(cur)
                fg = True
            else:
                if cur[0] > self.pqueue.peek()[0]:
                    self.pqueue.get()
                    self.pqueue.put(cur)
                if np.square(p[dim]-self.data[rt].feature[dim]) < -self.pqueue.peek()[0]:
                    fg = True
        if y != -1 and fg:
            self.query(p, m, y, dept+1)



