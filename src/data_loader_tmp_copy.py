import copy
import sys

import pandas as pd
import os


import utils
sys.path.append("../../")


class Database:
    def __init__(self, sub_dataset_name, task_name):
        self.sub_dataset_name = sub_dataset_name
        self.task_name = task_name

        # define path of file
        self.__project_root_path = utils.get_root_path_of_project("PathwayGNN")
        # self.__raw_data_file_path = os.path.join("data", sub_dataset_name)
        self.__raw_data_file_path = os.path.join("../data", sub_dataset_name)
        self.__task_file_path = os.path.join(self.__raw_data_file_path, task_name)


        # node mask
        self.__train_nodes_mask, self.__validation_nodes_mask, self.__test_nodes_mask = self.__get_nodes_mask()

        # node features
        self.__raw_nodes_features = self.__get_nodes_features_assist("raw")
        self.__train_nodes_features = self.__get_nodes_features_assist("train")
        self.__validation_nodes_features = self.__get_complete_nodes_features_mix_negative_for_attribute_prediction(self.__validation_nodes_mask, "validation")
        self.__test_nodes_features = self.__get_complete_nodes_features_mix_negative_for_attribute_prediction(self.__test_nodes_mask, "test")

        self.__function_dict = {"num_nodes": self.__get_num_of_nodes_based_on_type_name(),
                              "num_features": self.__get_num_of_features_based_on_type_name(),
                              "num_edges": self.__get_num_of_edges_based_on_type_name(),
                              "edge_list": self.__get_edge_of_nodes_list_regardless_direction(),
                              "raw_nodes_features": self.__raw_nodes_features,
                              "train_nodes_features": self.__train_nodes_features,
                              "validation_nodes_features": self.__validation_nodes_features,
                              "test_nodes_features": self.__test_nodes_features,
                              "train_node_mask": self.__train_nodes_mask,
                              "val_node_mask": self.__validation_nodes_mask,
                              "test_node_mask": self.__test_nodes_mask}

    def __getitem__(self, key):
        return self.__function_dict[key]

    def __get_num_of_nodes_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.__raw_data_file_path
        else:
            path: str = os.path.join(self.__task_file_path, type_name)
        node_line_message_list: list[str] = utils.read_file_via_lines(path, "nodes.txt")
        num_of_nodes = len(node_line_message_list)
        return num_of_nodes

    def __get_num_of_features_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.__raw_data_file_path
        else:
            path: str = os.path.join(self.__task_file_path, type_name)
        feature_line_message_list: list[str] = utils.read_file_via_lines(path, "components-all.txt")
        num_of_features = len(feature_line_message_list)
        return num_of_features

    def __get_num_of_edges_based_on_type_name(self, type_name: str = "raw") -> int:
        if "raw" == type_name:
            path: str = self.__raw_data_file_path
        else:
            path: str = os.path.join(self.__task_file_path, type_name)
        edge_line_message_list: list[str] = utils.read_file_via_lines(path, "edges.txt")
        num_of_edges = len(edge_line_message_list)
        return num_of_edges

    def __get_nodes_features_assist(self, type_name: str):
        if "raw" == type_name:
            path: str = self.__raw_data_file_path
        else:
            path: str = os.path.join(self.__task_file_path, type_name)

        num_of_nodes = self.__get_num_of_nodes_based_on_type_name(type_name)
        num_of_feature_dimension = self.__get_num_of_features_based_on_type_name()

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

        print(type_name + " dataset\n", "Number of interactions: %2d.\n Number of nodes: %2d.\n Number of features: %2d"
              % (len(mat), num_of_nodes, num_of_feature_dimension))

        return nodes_features

    def __get_complete_nodes_features_mix_negative_for_attribute_prediction(self, node_mask: list[int], type_name: str):
        nodes_features_mix_negative: list[list[int]] = self.__get_nodes_features_mix_negative_assist(type_name)

        path: str = self.__raw_data_file_path
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

        num_of_nodes = self.__get_num_of_nodes_based_on_type_name()
        num_of_feature_dimension = self.__get_num_of_features_based_on_type_name()

        nodes_features = utils.encode_node_features(raw_nodes_components_mapping_list, num_of_nodes, num_of_feature_dimension)

        return nodes_features

    def __get_nodes_features_mix_negative_assist(self, type_name: str) -> list[list[int]]:
        if "test" != type_name and "validation" != type_name:
            raise Exception("The type should be \"test\" or \"validation\"")
        if "attribute prediction dataset" != self.task_name:
            raise Exception(
                "The method \"self.__get_nodes_features_mix_negative_assist\" is only for attribute prediction task")
        path: str = os.path.join(self.__task_file_path, type_name)
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

    def __get_edge_of_nodes_list_regardless_direction(self) -> list[list[int]]:
        """
        :return: [[1,2,3], [3,7,9], [4,6,7,8,10,11]...] while [1,2,3], [3,7,9], .. represent the hyper edges
        """
        edge_to_list_of_nodes_dict, _, _ = self.__get_edge_to_list_of_nodes_dict("train")

        edge_of_nodes_list_without_direction: list[list[int]] = [list_of_nodes for edge_index, list_of_nodes in
                                                                 edge_to_list_of_nodes_dict.items()]

        return edge_of_nodes_list_without_direction

    def __get_edge_to_list_of_nodes_dict(self, type_name: str):
        """
        :param type_name: "raw" for raw dataset, "test" for test dataset, "train" for train dataset, "validation" for validation dataset
        :return:
        """
        if "raw" == type_name:
            path: str = self.__raw_data_file_path
        else:
            path: str = os.path.join(self.__task_file_path, type_name)

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

    def __get_nodes_mask(self) -> tuple[list[int], list[int], list[int]]:
        train_nodes_mask = self.__get_nodes_mask_assist("train")
        validation_nodes_mask = self.__get_nodes_mask_assist("validation")
        test_nodes_mask = self.__get_nodes_mask_assist("test")

        return train_nodes_mask, validation_nodes_mask, test_nodes_mask

    def __get_nodes_mask_assist(self, type_name: str) -> list[int]:
        nodes_mask: list[int] = list()

        path: str = os.path.join(self.__task_file_path, type_name)
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

    def __get_edges_mask_assist(self, type_name: str) -> list[int]:
        edges_mask: list[int] = list()

        path: str = os.path.join(self.__task_file_path, type_name)
        edges_line_message_list: list[str] = utils.read_file_via_lines(path, "edges.txt")

        for node_line_message in edges_line_message_list:
            elements = node_line_message.split(",")
            edge_index = int(elements[0])
            edges_mask.append(edge_index)

        return edges_mask


if __name__ == '__main__':
    # name = 'Disease'
    # task = 'attribute prediction dataset'
    # data_base = Database(name, task)
    # train, train_fea = data_base.train
    # print(train['entity'].tolist())
    data_loader = Database("Disease", "attribute prediction dataset")

    num = data_loader["num_nodes"]
    # validation_nodes_features

    validation_nodes_features = data_loader["validation_nodes_features"]

    print("validation_nodes_features: ", len(validation_nodes_features))

    # feature_test = [[1, 0, 1], [1, 1, 1], [1, 0, 0]]
    # feature_test = utils.get_normalized_features_in_tensor(feature_test)
    # print(feature_test)
