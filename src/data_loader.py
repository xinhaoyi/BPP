import os
import sys

import pandas as pd

sys.path.append("../src/")


class Database:
    """
    This is a dataloader for link prediction dataset
    Args:
            name (string): Name of the dataset e.g. Disease.
            task (string): Name of the task e.g. input link preidction dataset
    Return:
            self.train/test/valid (df): Dataframe of train/test/valid sets.

    """

    def __init__(self, name, task):
        self.dataset = name
        self.task = task
        self.load_dataset()

    def load_dataset(self):
        self.train = self.load_train_to_graph(self.dataset, self.task, "train")
        self.test = self.load_other_to_graph(self.dataset, self.task, "test")
        self.valid = self.load_other_to_graph(self.dataset, self.task, "validation")

    def load_train_to_graph(self, name, task, subset):
        data_path = os.path.join("data", name)
        relation_path = os.path.join(data_path, task, subset, "relationship.txt")
        mapping_path = os.path.join(data_path, task, subset, "components-mapping.txt")
        mat = pd.read_csv(
            relation_path, names=["entity", "reaction", "type"], header=None
        )
        my_file = open(mapping_path, "r")
        mapping = my_file.read()
        mapping_list = mapping.split("\n")
        new_list = [i.split(",") for i in mapping_list]
        final_list = []
        for i in new_list:
            final_list.append([int(j) for j in i])
        # mapping = pd.read_csv(train_mapping_path)
        feature_dimension = max(sum(final_list, [])) + 1
        num_nodes = max(mat["entity"])
        print(
            subset,
            "Num of interactions: %2d.\n Number of nodes: %2d.\n Number of features: %2d"
            % (len(mat), num_nodes, feature_dimension),
        )
        return mat

    def load_other_to_graph(self, name, task, subset):
        data_path = os.path.join("data", name)
        relation_path = os.path.join(data_path, task, subset, "relationship.txt")
        mat = pd.read_csv(
            relation_path, names=["entity", "reaction", "type"], header=None
        )
        print("Load %s set" % subset)
        return mat
