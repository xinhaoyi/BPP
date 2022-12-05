import copy
import os
import pandas as pd
import sys

import utils

sys.path.append("../src/")


class Database:
    """
    This is a dataloader for link prediction dataset
    Args:
            name (string): Name of the dataset e.g. Disease.
            task (string): Name of the task e.g. input link prediction dataset
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
        data_path = os.path.join("../data", name)
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
        data_path = os.path.join("../data", name)
        relation_path = os.path.join(data_path, task, subset, "relationship.txt")
        mat = pd.read_csv(
            relation_path, names=["entity", "reaction", "type"], header=None
        )
        print("Load %s set" % subset)
        return mat


class DataLoaderBase:
    def __init__(self, sub_dataset_name, task_name):
        self.sub_dataset_name = sub_dataset_name
        self.task_name = task_name

        # define path of file
        # self.__raw_data_file_path = os.path.join("data", sub_dataset_name)
        self.raw_data_file_path = os.path.join("..", "data", sub_dataset_name)
        self.task_file_path = os.path.join(self.raw_data_file_path, task_name)

    def get_num_of_nodes_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        node_line_message_list: list[str] = utils.read_file_via_lines(path, "nodes.txt")
        num_of_nodes = len(node_line_message_list)
        return num_of_nodes

    def get_num_of_features_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        feature_line_message_list: list[str] = utils.read_file_via_lines(path, "components-all.txt")
        num_of_features = len(feature_line_message_list)
        return num_of_features

    def get_num_of_edges_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)
        edge_line_message_list: list[str] = utils.read_file_via_lines(path, "edges.txt")
        num_of_edges = len(edge_line_message_list)
        return num_of_edges

    def get_nodes_features_assist(self, type_name: str):
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)

        num_of_nodes = self.get_num_of_nodes_based_on_type_name(type_name)
        num_of_edges = self.get_num_of_edges_based_on_type_name(type_name)
        num_of_feature_dimension = self.get_num_of_features_based_on_type_name()

        relationship_path = os.path.join(path, "relationship.txt")
        # mat = pd.read_csv(os.path.join(self.__project_root_path, relationship_path), names=['entity', 'reaction', 'type'], header=None)
        mat = pd.read_csv(relationship_path, names=['entity', 'reaction', 'type'], header=None)

        components_mapping_line_message_list: list[str] = utils.read_file_via_lines(path, "components-mapping.txt")
        components_mapping_list_with_str_style = [components_mapping_line_message.split(',') for
                                                  components_mapping_line_message in
                                                  components_mapping_line_message_list]

        components_mapping_list = []

        for components_mapping_str in components_mapping_list_with_str_style:
            components_mapping_line_int_style = [int(component) for component in components_mapping_str]
            components_mapping_list.append(components_mapping_line_int_style)

        nodes_features = utils.encode_node_features(components_mapping_list, num_of_nodes, num_of_feature_dimension)

        print(type_name + " dataset\n", "Number of interactions: %2d.\n Number of nodes: %2d.\n Number of features: %2d.\n Number of edges: %2d."
              % (len(mat), num_of_nodes, num_of_feature_dimension, num_of_edges))

        return nodes_features

    def get_edge_of_nodes_list_regardless_direction(self, param) -> list[list[int]]:
        """
        Get the nodes of all the hyper edges
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """
        pass

    def get_edge_to_list_of_nodes_dict(self, type_name: str):
        """
        :param type_name: "raw" for raw dataset, "test" for test dataset, "train" for train dataset, "validation" for validation dataset
        :return:
        """
        if "raw" == type_name:
            path: str = self.raw_data_file_path
        else:
            path: str = os.path.join(self.task_file_path, type_name)

        relationship_line_message_list: list[str] = utils.read_file_via_lines(path, "relationship.txt")

        edge_to_list_of_nodes_dict: dict[int, list[int]] = dict()
        edge_to_list_of_input_nodes_dict: dict[int, list[int]] = dict()
        edge_to_list_of_output_nodes_dict: dict[int, list[int]] = dict()

        for relationship_line_message in relationship_line_message_list:
            elements: list[str] = relationship_line_message.split(",")
            node_index: int = int(elements[0])
            edge_index: int = int(elements[1])
            direction: int = int(elements[2])

            if edge_index not in edge_to_list_of_nodes_dict.keys():
                edge_to_list_of_nodes_dict[edge_index] = list()
            edge_to_list_of_nodes_dict[edge_index].append(node_index)

            if direction < 0:
                if edge_index not in edge_to_list_of_input_nodes_dict.keys():
                    edge_to_list_of_input_nodes_dict[edge_index] = list()
                edge_to_list_of_input_nodes_dict[edge_index].append(node_index)

            elif direction > 0:
                if edge_index not in edge_to_list_of_output_nodes_dict.keys():
                    edge_to_list_of_output_nodes_dict[edge_index] = list()
                edge_to_list_of_output_nodes_dict[edge_index].append(node_index)

        return edge_to_list_of_nodes_dict, edge_to_list_of_input_nodes_dict, edge_to_list_of_output_nodes_dict

    def get_nodes_mask_assist(self, type_name: str) -> list[int]:
        nodes_mask: list[int] = list()

        path: str = os.path.join(self.task_file_path, type_name)
        if "train" != type_name:
            file_name = "nodes.txt"
        else:
            file_name = "nodes-mask.txt"

        node_line_message_list: list[str] = utils.read_file_via_lines(path, file_name)

        for node_line_message in node_line_message_list:
            elements = node_line_message.split(",")
            node_index = int(elements[0])
            nodes_mask.append(node_index)

        return nodes_mask

    def get_edges_mask_assist(self, type_name: str) -> list[int]:
        edges_mask: list[int] = list()

        path: str = os.path.join(self.task_file_path, type_name)
        edges_line_message_list: list[str] = utils.read_file_via_lines(path, "edges.txt")

        for node_line_message in edges_line_message_list:
            elements = node_line_message.split(",")
            edge_index = int(elements[0])
            edges_mask.append(edge_index)

        return edges_mask


class DataLoaderAttribute(DataLoaderBase):
    def __init__(self, sub_dataset_name, task_name):
        super().__init__(sub_dataset_name, task_name)

        # node mask
        self.__train_nodes_mask, self.__validation_nodes_mask, self.__test_nodes_mask = self.__get_nodes_mask()

        # node features
        self.__raw_nodes_features = self.get_nodes_features_assist("raw")
        self.__train_nodes_features = self.get_nodes_features_assist("train")
        self.__validation_nodes_features = self.__get_complete_nodes_features_mix_negative_for_attribute_prediction(
            self.__validation_nodes_mask, "validation")
        self.__test_nodes_features = self.__get_complete_nodes_features_mix_negative_for_attribute_prediction(
            self.__test_nodes_mask, "test")

        self.__function_dict = {"num_nodes": self.get_num_of_nodes_based_on_type_name(),
                                "num_features": self.get_num_of_features_based_on_type_name(),
                                "num_edges": self.get_num_of_edges_based_on_type_name(),
                                "edge_list": self.get_edge_of_nodes_list_regardless_direction("train"),
                                "raw_nodes_features": self.__raw_nodes_features,
                                "train_nodes_features": self.__train_nodes_features,
                                "validation_nodes_features": self.__validation_nodes_features,
                                "test_nodes_features": self.__test_nodes_features,
                                "train_node_mask": self.__train_nodes_mask,
                                "val_node_mask": self.__validation_nodes_mask,
                                "test_node_mask": self.__test_nodes_mask}

    def __getitem__(self, key):
        return self.__function_dict[key]

    def __get_complete_nodes_features_mix_negative_for_attribute_prediction(self, node_mask: list[int], type_name: str):
        nodes_features_mix_negative: list[list[int]] = self.__get_nodes_features_mix_negative_assist(type_name)

        path: str = self.raw_data_file_path
        components_mapping_line_message_list: list[str] = utils.read_file_via_lines(path, "components-mapping.txt")
        components_mapping_list_with_str_style = [components_mapping_line_message.split(',') for
                                                  components_mapping_line_message in
                                                  components_mapping_line_message_list]

        raw_nodes_components_mapping_list = []

        for components_mapping_str in components_mapping_list_with_str_style:
            components_mapping_line_int_style = [int(component) for component in components_mapping_str]
            raw_nodes_components_mapping_list.append(components_mapping_line_int_style)

        for i, node_mask_index in enumerate(node_mask):
            raw_nodes_components_mapping_list[node_mask_index] = nodes_features_mix_negative[i]

        num_of_nodes = self.get_num_of_nodes_based_on_type_name()
        num_of_feature_dimension = self.get_num_of_features_based_on_type_name()

        nodes_features = utils.encode_node_features(raw_nodes_components_mapping_list, num_of_nodes,
                                                    num_of_feature_dimension)

        return nodes_features

    def __get_nodes_features_mix_negative_assist(self, type_name: str) -> list[list[int]]:
        if "test" != type_name and "validation" != type_name:
            raise Exception("The type should be \"test\" or \"validation\"")
        if "attribute prediction dataset" != self.task_name:
            raise Exception(
                "The method \"self.__get_nodes_features_mix_negative_assist\" is only for attribute prediction task")
        path: str = os.path.join(self.task_file_path, type_name)
        components_mapping_line_message_mix_negative_list: list[str] = utils.read_file_via_lines(path,
                                                                                                 "components-mapping-mix-negative.txt")

        nodes_features_mix_negative: list[list[int]] = list()

        for components_mapping_line_message_mix_negative in components_mapping_line_message_mix_negative_list:
            elements: list[str] = components_mapping_line_message_mix_negative.split("||")
            positive_components_list_str_message: str = elements[0]
            negative_components_list_str_style: list[str] = elements[1:-1]

            components_list: list[int] = list()

            positive_components_list: list[int] = [int(positive_component_str_style) for positive_component_str_style in
                                                   positive_components_list_str_message.split(",")]
            negative_components_list: list[int] = [int(negative_components_str_style) for negative_components_str_style
                                                   in negative_components_list_str_style]

            components_list.extend(positive_components_list)

            components_list.extend(negative_components_list)

            nodes_features_mix_negative.append(copy.deepcopy(components_list))

        return nodes_features_mix_negative

    def __get_nodes_mask(self) -> tuple[list[int], list[int], list[int]]:
        train_nodes_mask = super().get_nodes_mask_assist("train")
        validation_nodes_mask = super().get_nodes_mask_assist("validation")
        test_nodes_mask = super().get_nodes_mask_assist("test")

        return train_nodes_mask, validation_nodes_mask, test_nodes_mask

    def get_edge_of_nodes_list_regardless_direction(self, type_name: str) -> list[list[int]]:
        """
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """
        edge_to_list_of_nodes_dict, _, _ = self.get_edge_to_list_of_nodes_dict(type_name)

        edge_of_nodes_list_without_direction: list[list[int]] = [list_of_nodes for edge_index, list_of_nodes in
                                                                 edge_to_list_of_nodes_dict.items()]

        return edge_of_nodes_list_without_direction


class DataLoaderLink(DataLoaderBase):
    def __init__(self, sub_dataset_name, task_name):
        super().__init__(sub_dataset_name, task_name)

        self.__raw_edge_to_nodes_dict, self.__train_edge_to_nodes_dict, self.__validation_edge_to_nodes_dict, self.__test_edge_to_nodes_dict = self.__get_edge_to_list_of_nodes_dict()
        self.__train_edge_mask, self.__validation_edge_mask, self.__test_edge_mask = self.get_edges_mask()

        self.__raw_nodes_features, self.__train_nodes_features, self.__validation_nodes_features, self.__test_nodes_features = (
            super().get_nodes_features_assist(
                "raw"), super().get_nodes_features_assist("train"), super().get_nodes_features_assist(
                "validation"), super().get_nodes_features_assist("test"))

        self.__function_dict = {"num_nodes": self.get_num_of_nodes_based_on_type_name(),
                                "num_features": self.get_num_of_features_based_on_type_name(),
                                "num_edges": self.get_num_of_edges_based_on_type_name(),
                                "raw_edge_list": self.get_edge_of_nodes_list_regardless_direction("raw"),
                                "train_edge_list": self.get_edge_of_nodes_list_regardless_direction("train"),
                                "validation_edge_list": self.get_edge_of_nodes_list_regardless_direction("validation"),
                                "test_edge_list": self.get_edge_of_nodes_list_regardless_direction("test"),
                                "raw_nodes_features": self.__raw_nodes_features,
                                "train_nodes_features": self.__train_nodes_features,
                                "validation_nodes_features": self.__validation_nodes_features,
                                "test_nodes_features": self.__test_nodes_features,
                                "train_edge_mask": self.__train_edge_mask,
                                "val_edge_mask": self.__validation_edge_mask,
                                "test_edge_mask": self.__test_edge_mask}

    def __getitem__(self, key):
        return self.__function_dict[key]

    def __get_edge_to_list_of_nodes_dict(self) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
        train_edge_to_list_of_nodes_dict, _, _ = super().get_edge_to_list_of_nodes_dict("train")
        validation_edge_to_list_of_nodes_dict, _, _ = super().get_edge_to_list_of_nodes_dict(
            "validation")
        test_edge_to_list_of_nodes_dict, _, _ = super().get_edge_to_list_of_nodes_dict("test")

        raw_edge_to_list_of_nodes_dict_complete, _, _ = self.get_edge_to_list_of_nodes_dict("raw")
        train_edge_to_list_of_nodes_dict_complete, _, _ = self.get_edge_to_list_of_nodes_dict("raw")
        validation_edge_to_list_of_nodes_dict_complete, _, _ = self.get_edge_to_list_of_nodes_dict("raw")
        test_edge_to_list_of_nodes_dict_complete, _, _ = self.get_edge_to_list_of_nodes_dict("raw")

        for edge, list_of_nodes in train_edge_to_list_of_nodes_dict.items():
            train_edge_to_list_of_nodes_dict_complete[edge] = list_of_nodes

        for edge, list_of_nodes in validation_edge_to_list_of_nodes_dict.items():
            validation_edge_to_list_of_nodes_dict_complete[edge] = list_of_nodes

        for edge, list_of_nodes in test_edge_to_list_of_nodes_dict.items():
            test_edge_to_list_of_nodes_dict_complete[edge] = list_of_nodes

        return raw_edge_to_list_of_nodes_dict_complete, train_edge_to_list_of_nodes_dict_complete, validation_edge_to_list_of_nodes_dict_complete, test_edge_to_list_of_nodes_dict_complete

    def get_edge_of_nodes_list_regardless_direction(self, type_name) -> list[list[int]]:
        """
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """

        type_dict = {"raw": self.__raw_edge_to_nodes_dict,
                     "train": self.__train_edge_to_nodes_dict,
                     "validation": self.__validation_edge_to_nodes_dict,
                     "test": self.__test_edge_to_nodes_dict}

        if type_name not in type_dict.keys():
            raise Exception("Please input \"train\", \"validation\" or \"test\" ")

        edge_of_nodes_list_without_direction: list[list[int]] = [list_of_nodes for edge_index, list_of_nodes in
                                                                 type_dict[type_name].items()]

        return edge_of_nodes_list_without_direction

    def get_edges_mask(self):
        validation_edges_mask = super().get_edges_mask_assist("validation")
        test_edges_mask = super().get_edges_mask_assist("test")

        train_edges_set = set()
        for validation_edges_idx in validation_edges_mask:
            train_edges_set.add(validation_edges_idx)
        for test_edges_idx in test_edges_mask:
            train_edges_set.add(test_edges_idx)

        train_edges_mask = list(train_edges_set)

        train_edges_mask.sort()

        return train_edges_mask, validation_edges_mask, test_edges_mask


if __name__ == '__main__':
    # name = 'Disease'
    # task = 'attribute prediction dataset'
    # data_base = Database(name, task)
    # train, train_fea = data_base.train
    # print(train['entity'].tolist())
    data_loader = DataLoaderAttribute("Disease", "attribute prediction dataset")

    num = data_loader["num_nodes"]
    # validation_nodes_features

    validation_nodes_features = data_loader["validation_nodes_features"]

    print("validation_nodes_features: ", len(validation_nodes_features))

    # feature_test = [[1, 0, 1], [1, 1, 1], [1, 0, 0]]
    # feature_test = utils.get_normalized_features_in_tensor(feature_test)
    # print(feature_test)
