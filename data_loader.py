import pandas as pd
import os
import sys
from scipy.sparse import csr_matrix
sys.path.append("../")


name = 'Disease'
task = 'input link prediction dataset'

class Database():
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
        feature_dimension = max(sum(final_list, []))+1
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
              %(len(mat), num_nodes, feature_dimension))
        return mat, component_csc_mat

data = Database(name,task)
train, train_fea = data.train
test, test_fea = data.test
valid, valid_fea = data.valid
