import numpy as np
import pandas as pd
import os
import sys

import scipy.sparse as sp
import torch
from numpy import ndarray
from scipy.sparse import csr_matrix, coo_matrix

sys.path.append("../")
name = 'Disease'
task = 'input link prediction dataset'


# todo 发现一个尚未解决的bug，即对于 attribute prediction，train中的relationships数量和raw data中relationships数量不相等，差2个
# todo 初步检查是因为raw data 中relationship的去重还存在问题，导致在变成train data去重时，删掉了两个重复的，这个bug还没有修复
class Database:
    def __init__(self, name, task):
        self.dataset = name
        self.task = task
        self.load_dataset()

    def load_dataset(self):
        self.train = self.load_data_to_graph(self.dataset, self.task, 'train')
        self.test = self.load_data_to_graph(self.dataset, self.task, 'test')
        self.valid = self.load_data_to_graph(self.dataset, self.task, 'validation')

    def load_data_to_graph(self, name, task, subset):
        data_path = os.path.join("data", name)

        raw_relation_path = os.path.join(data_path, 'relationship.txt')


        relation_path = os.path.join(data_path, task, subset, 'relationship.txt')
        mapping_path = os.path.join(data_path, task, subset, 'components-mapping.txt')

        mat = pd.read_csv(relation_path, names=['entity', 'reaction', 'type'], header=None)
        my_file = open(mapping_path, "r")
        mapping = my_file.read()
        mapping_list = mapping.split("\n")
        new_list = [i.split(',') for i in mapping_list]
        final_list = []

        for i in new_list:
            final_list.append([int(j) for j in i])

        # mapping = pd.read_csv(train_mapping_path)
        feature_dimension = max(sum(final_list, [])) + 1
        num_nodes = max(mat['entity'])

        row = []
        column = []
        val = []
        for i in range(num_nodes):
            feature = final_list[i]
            for j in feature:
                row.append(i)
                column.append(j)
                val.append(1)

        component_csc_mat = csr_matrix((val, (row, column)), shape=(num_nodes, feature_dimension))
        print(subset, "Num of interactions: %2d.\n Number of nodes: %2d.\n Number of features: %2d"
              % (len(mat), num_nodes, feature_dimension))
        return mat, component_csc_mat

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        """
        numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
        pytorch中的tensor转化成numpy中的ndarray : numpy()
        """

        # tocoo() convert COO形式
        # 将稀疏矩阵转化为COO格式，矩阵矢量乘积这一数值计算中经常用的操作会变得非常高效。
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

        # indices are the coordinates of the adjacency matrix
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

        #
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.FloatTensor(indices, values, shape)

        # https://ptorch.com/docs/1/torch-sparse
        # Don't worry about the yellow report, it's actually a bug in pycharm
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_normalized_features_in_tensor(self, features) -> torch.Tensor:
        features_mat: csr_matrix = csr_matrix(features, dtype=np.float32)
        features_mat: csr_matrix = self.normalize_sparse_matrix(features_mat)
        features: torch.Tensor = torch.FloatTensor(np.array(features_mat.todense()))
        return features

    @staticmethod
    def normalize_sparse_matrix(mat):
        """Row-normalize sparse matrix"""
        # sum(1) 是计算每一行的和
        # 会得到一个（2708,1）的矩阵
        rowsum: ndarray = np.array(mat.sum(1))

        # 把这玩意儿取倒，然后拍平
        r_inv = np.power(rowsum, -1).flatten()

        # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
        r_inv[np.isinf(r_inv)] = 0.

        # np.diag() 应该也可以
        # 这里就是生成 对角矩阵
        r_mat_inv = sp.diags(r_inv)

        # 点乘,得到归一化后的结果
        # 注意是 归一化矩阵 点乘 原矩阵，别搞错了!!
        mat = r_mat_inv.dot(mat)
        return mat

    def get_list_of_edges_represent_via_nodes(self):
        train, train_fea = self.train
        test, test_fea = self.test
        valid, valid_fea = self.valid


# data = Database(name,task)
# train, train_fea = data.train
# test, test_fea = data.test
# valid, valid_fea = data.valid


if __name__ == '__main__':
    name = 'Disease'
    task = 'attribute prediction dataset'
    data_base = Database(name, task)
    train, train_fea = data_base.train
    print(train['entity'].tolist())

    feature_test = [[1, 0, 1], [1, 1, 1], [1, 0, 0]]
    feature_test = data_base.get_normalized_features_in_tensor(feature_test)
    print(feature_test)
