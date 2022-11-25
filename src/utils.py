import os
import platform

import numpy as np
import scipy as sp
import torch
from numpy import ndarray
from scipy.sparse import csr_matrix


def read_file_via_lines(path: str, file_name: str) -> list[str]:
    root_path: str = get_root_path_of_project("PathwayGNN")
    url: str = os.path.join(root_path, path, file_name)
    # url: str = os.path.join(path, file_name)
    res_list: list[str] = []

    try:
        file_handler = open(url, "r")
        while True:
            # Get next line from file
            line = file_handler.readline()
            line = line.replace('\r', '').replace('\n', '').replace('\t', '')

            # If the line is empty then the end of file reached
            if not line:
                break
            res_list.append(line)
    except Exception as e:
        print(e)
        print("we can't find the " + url + ", please make sure that the file exists")
    finally:
        return res_list

def get_sys_platform():
    sys_platform = platform.platform()
    if "Windows" in sys_platform:
        sys_platform_return = "windows"
    elif "macOS" in sys_platform:
        sys_platform_return = "macos"
    elif "linux" in sys_platform:
        sys_platform_return = "linux"
    else:
        sys_platform_return = "other"
    return sys_platform_return

def get_root_path_of_project(project_name: str):
    """
    This method is to get the root path of the project
    ex. when project name is PathwayGNN, root path is "E:\Python_Project\HGNN_On_Reactome\PathwayGNN"
    :param project_name: the name of the project
    :return:
    """
    cur_path: str = os.path.abspath(os.path.dirname(__file__))
    if "windows" == get_sys_platform():
        root_path: str = cur_path[:cur_path.find(project_name + "\\") + len(project_name + "\\")]
    elif "macos" == get_sys_platform() or "linux" == get_sys_platform():
        root_path: str = cur_path[:cur_path.find(project_name + "/") + len(project_name + "/")]
    else:
        raise Exception("We can't support other system platform! Please use windows or macos")
    return root_path


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


def get_normalized_features_in_tensor(features) -> torch.Tensor:
    features_mat: csr_matrix = csr_matrix(features, dtype=np.float32)
    features_mat: csr_matrix = normalize_sparse_matrix(features_mat)
    features: torch.Tensor = torch.FloatTensor(np.array(features_mat.todense()))
    return features


def encode_node_features(components_mapping_list: list[list[int]], num_of_nodes: int, num_of_feature_dimension: int) -> list[list[int]]:
    row = []
    column = []
    val = []
    for line_index in range(num_of_nodes):
        features_of_one_entity = components_mapping_list[line_index]
        for feature_index in features_of_one_entity:
            row.append(line_index)
            column.append(feature_index)
            val.append(1)

    component_csc_mat = csr_matrix((val, (row, column)), shape=(num_of_nodes, num_of_feature_dimension))
    nodes_features: list[list[int]] = component_csc_mat.toarray().tolist()

    return nodes_features
