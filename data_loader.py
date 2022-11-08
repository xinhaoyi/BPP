import pandas as pd
import numpy as np
import os
import sys
import time
from scipy.sparse import csc_array
sys.path.append("../")



name = 'Disease'
task = 'attribute prediction dataset'

def load_data_to_graph(name, task):
    data_path = os.path.join("data", name)
    train_relation_path = os.path.join(data_path, task, 'train', 'relationship.txt')
    train_node_path = os.path.join(data_path, task, 'train', 'nodes.txt')
    train_mapping_path = os.path.join(data_path, task, 'train', 'components-mapping.txt')
    train_mat = pd.read_csv(train_relation_path, names=['entity', 'reaction', 'type'], header=None)
    # node = pd.read_csv(train_node_path)
    my_file = open(train_mapping_path, "r")
    mapping = my_file.read()
    mapping_list = mapping.split("\n")
    new_list = [i.split(',') for i in mapping_list]
    final_list = []
    for i in new_list:
        if len(i[0]) == 0:
            final_list.append([])
        else:
            final_list.append([int(j) for j in i])
    # mapping = pd.read_csv(train_mapping_path)
    feature_dimension = max(final_list)
    feature_dimension = max(feature_dimension)
    num_nodes = max(train_mat['entity'])

    row = []
    column = []
    val = []
    for i in range(num_nodes):
        feature = final_list[i]
        if len(feature) > 0:
            for j in feature:
                row.append(i)
                column.append(j)
                val.append(1)
    csc_mat = csc_array((val, (row, column)), shape=(num_nodes, feature_dimension))
    return train_mat, csc_mat

# To recover the numpy matrix of feature matrix, use feature_mat.toarray()
interaction_mat, feature_mat = load_data_to_graph(name=name, task=task)
