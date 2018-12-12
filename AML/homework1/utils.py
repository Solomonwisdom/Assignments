# -*- coding: utf-8 -*-

import numpy as np
import queue


class PeekablePriorityQueue(queue.PriorityQueue):
    def peek(self):
        """Peeks the first element of the queue

        Returns
        -------
        item : object
            First item in the queue

        Raises
        ------
        queue.Empty
            No items in the queue
        """
        try:
            with self.mutex:
                return self.queue[0]
        except IndexError:
            raise queue.Empty


def load_dataset(filename):
    X = []
    Y = []
    with open(filename, "r") as f:
        for line in f.readlines():
            nums = line.strip().split(",")
            X.append([float(x) for x in nums[:-1]])
            Y.append(int(nums[-1]))
    return np.array(X), np.array(Y)


def get_distance(data):
    # 获取欧氏距离
    # data: 要获取欧氏距离的矩阵，大小 m * n
    # return：m * m 的矩阵，第 [i, j] 个元素代表 data 中元素 i 到元素 j 的欧氏距离
    data_count = data.shape[0]
    distance_mat = np.zeros([data_count, data_count], np.float32)
    for idx in range(data_count):
        distance_mat[idx] = np.array(
            np.linalg.norm(np.tile(data[idx], (data_count, 1)) - data, axis=1)).reshape((data_count,))
    return distance_mat
