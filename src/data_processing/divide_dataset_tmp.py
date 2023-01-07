from __future__ import annotations

import copy
import os
import random
import re
from enum import Enum

import numpy as np
from extract_data_form_reactome import FileProcessor
from property import Properties


class Attribute:
    def __init__(self, index: int, stId: str, name: str, dataset_type: str):
        self.index: int = index
        self.stId: str = stId
        self.name: str = name
        self.dataset_type: str = dataset_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Attribute):
            return (self.index == o.index) and (self.stId == o.stId) and (self.name == o.name) and (
                    self.dataset_type == o.dataset_type)

    def __hash__(self) -> int:
        return hash(self.index) + hash(self.stId) + hash(self.name) + hash(self.dataset_type)


class Node:
    def __init__(self, index: int, stId: str, name: str, dataset_type: str):
        self.index: int = index
        self.stId: str = stId
        self.name: str = name
        self.dataset_type: str = dataset_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Attribute):
            return (self.index == o.index) and (self.stId == o.stId) and (self.name == o.name) and (
                    self.dataset_type == o.dataset_type)

    def __hash__(self) -> int:
        return hash(self.index) + hash(self.stId) + hash(self.name) + hash(self.dataset_type)


class Edge:
    def __init__(self, index: int, stId: str, name: str, dataset_type: str):
        self.index: int = index
        self.stId: str = stId
        self.name: str = name
        self.dataset_type: str = dataset_type

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Attribute):
            return (self.index == o.index) and (self.stId == o.stId) and (self.name == o.name) and (
                    self.dataset_type == o.dataset_type)
        return False

    def __hash__(self) -> int:
        return hash(self.index) + hash(self.stId) + hash(self.name) + hash(self.dataset_type)


class PairOfNodeAndComponent:
    def __init__(self, node: Node, attribute: Attribute):
        self.node: Node = node
        self.attribute: Attribute = attribute

    def __eq__(self, o: object) -> bool:
        if isinstance(o, PairOfNodeAndComponent):
            return (self.node == o.node) and (self.attribute == o.attribute)
        return False

    def __hash__(self) -> int:
        return hash(self.node) + hash(self.attribute)


class Relationship:
    def __init__(self, node: Node, edge: Edge, direction: int):
        self.node: Node = node
        self.edge: Edge = edge
        self.direction: int = direction

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Relationship):
            return (self.node == o.node) and (self.edge == o.edge) and (self.direction == o.direction)
        return False

    def __hash__(self) -> int:
        return hash(self.node) + hash(self.edge) + hash(self.direction)


class DataSet:
    def __init__(self):
        self.list_of_attributes: set[Attribute] = set()
        self.list_of_nodes: set[Node] = set()
        self.list_of_edges: set[Edge] = set()
        self.list_of_pair_of_node_and_component: set[PairOfNodeAndComponent] = set()
        self.list_of_relationship: set[Relationship] = set()

        self.dataset_message = DataSetTextMessage()

        self.dataset_obs = DataSetObs()

    def add_attribute(self, attribute: Attribute):
        if isinstance(attribute, Attribute):
            self.list_of_attributes.add(attribute)

            # notify the pair_of_node_and_component_obs
            self.dataset_obs.pair_of_node_and_component_obs.update_add(attribute)

    def add_node(self, node: Node):
        if isinstance(node, Node):
            self.list_of_nodes.add(node)

            # notify the pair_of_node_and_component_obs
            self.dataset_obs.pair_of_node_and_component_obs.update_add(node)

            # notify the relationship_obs
            self.dataset_obs.relationship_obs.update_add(node)

    def add_edge(self, edge: Edge):
        if isinstance(edge, Edge):
            self.list_of_edges.add(edge)

            # notify the relationship_obs
            self.dataset_obs.relationship_obs.update_add(edge)

    def add_pair_of_node_and_component(self, pair_of_node_and_component: PairOfNodeAndComponent):
        self.list_of_pair_of_node_and_component.add(pair_of_node_and_component)

        # notify the pair_of_node_and_component_obs
        self.dataset_obs.pair_of_node_and_component_obs.update_add(pair_of_node_and_component)

    def add_relationship(self, relationship: Relationship):
        self.list_of_relationship.add(relationship)

        # notify the relationship_obs
        self.dataset_obs.relationship_obs.update_add(relationship)

    def delete_pair_of_node_and_component(self, pair_of_node_and_component: PairOfNodeAndComponent):
        self.list_of_pair_of_node_and_component.remove(pair_of_node_and_component)

        # notify the pair_of_node_and_component_obs
        self.dataset_obs.pair_of_node_and_component_obs.update_delete(pair_of_node_and_component)

    def delete_relationship(self, relationship: Relationship):
        self.list_of_relationship.remove(relationship)

        # notify the relationship_obs
        self.dataset_obs.relationship_obs.update_delete(relationship)


class DataSetTextMessage:
    def __init__(self):
        self.pathway_name: str = ""
        # "attribute prediction dataset", "input link prediction dataset", "output link prediction dataset"
        self.task_name: str = ""
        # "train", "validation", "test"
        self.type_name: str= ""

    def initialize(self, **args):
        if "pathway_name" in args.keys():
            self.pathway_name = args["pathway_name"]
        else:
            raise Exception("pathway_name is needed!")

        if "task_name" in args.keys():
            self.task_name = args["task_name"]
        else:
            self.task_name = "raw data"

        if "type_name" in args.keys():
            self.type_name = args["type_name"]
        else:
            self.type_name = "raw"


class DataSetObs:
    class __PairOfNodeAndComponentObs:
        def __init__(self):
            self.node_to_set_of_attributes_dict: dict[Node, set[Attribute]] = dict()
            self.attribute_to_set_of_nodes_dict: dict[Attribute, set[Node]] = dict()

        def update_add(self, arg: Attribute | Node | PairOfNodeAndComponent):
            if isinstance(arg, Attribute):
                if arg not in self.attribute_to_set_of_nodes_dict.keys():
                    self.attribute_to_set_of_nodes_dict[arg] = set()

            if isinstance(arg, Node):
                if arg not in self.node_to_set_of_attributes_dict.keys():
                    self.node_to_set_of_attributes_dict[arg] = set()

            if isinstance(arg, PairOfNodeAndComponent):
                node = arg.node
                attribute = arg.attribute

                if node not in self.node_to_set_of_attributes_dict.keys():
                    self.node_to_set_of_attributes_dict[node] = set()
                self.node_to_set_of_attributes_dict[node].add(attribute)

                if attribute not in self.attribute_to_set_of_nodes_dict.keys():
                    self.attribute_to_set_of_nodes_dict[attribute] = set()
                self.attribute_to_set_of_nodes_dict[attribute].add(node)

        def update_delete(self, arg: PairOfNodeAndComponent):
            node = arg.node
            attribute = arg.attribute
            self.node_to_set_of_attributes_dict[node].remove(attribute)
            self.attribute_to_set_of_nodes_dict[attribute].remove(node)

    class __RelationshipObs:
        def __init__(self):
            self.edge_to_set_of_input_nodes_dict: dict[Edge, set[Node]] = dict()
            self.edge_to_set_of_output_nodes_dict: dict[Edge, set[Node]] = dict()
            self.edge_to_set_of_nodes_dict: dict[Edge, set[Node]] = dict()

            self.node_to_set_of_input_edges_dict: dict[Node, set[Edge]] = dict()
            self.node_to_set_of_output_edges_dict: dict[Node, set[Edge]] = dict()
            self.node_to_set_of_edges_dict: dict[Node, set[Edge]] = dict()

            self.edge_to_set_of_regulation_nodes_dict: dict[Edge, set[Node]] = dict()
            self.node_to_set_of_regulation_edges_dict: dict[Node, set[Edge]] = dict()


        def update_add(self, arg: Node | Edge | Relationship):
            if isinstance(arg, Node):
                if arg not in self.node_to_set_of_input_edges_dict.keys():
                    self.node_to_set_of_input_edges_dict[arg] = set()
                if arg not in self.node_to_set_of_output_edges_dict.keys():
                    self.node_to_set_of_output_edges_dict[arg] = set()
                if arg not in self.node_to_set_of_edges_dict.keys():
                    self.node_to_set_of_edges_dict[arg] = set()

                if arg not in self.node_to_set_of_regulation_edges_dict.keys():
                    self.node_to_set_of_regulation_edges_dict[arg] = set()

            if isinstance(arg, Edge):
                if arg not in self.edge_to_set_of_input_nodes_dict.keys():
                    self.edge_to_set_of_input_nodes_dict[arg] = set()
                if arg not in self.edge_to_set_of_output_nodes_dict.keys():
                    self.edge_to_set_of_output_nodes_dict[arg] = set()
                if arg not in self.edge_to_set_of_nodes_dict.keys():
                    self.edge_to_set_of_nodes_dict[arg] = set()

                if arg not in self.edge_to_set_of_regulation_nodes_dict.keys():
                    self.edge_to_set_of_regulation_nodes_dict[arg] = set()

            if isinstance(arg, Relationship):
                node = arg.node
                edge = arg.edge
                direction = arg.direction

                self.update_add(node)
                self.update_add(edge)

                self.edge_to_set_of_nodes_dict[edge].add(node)
                self.node_to_set_of_edges_dict[node].add(edge)

                if -1 == direction:
                    self.edge_to_set_of_input_nodes_dict[edge].add(node)
                    self.node_to_set_of_input_edges_dict[node].add(edge)

                if 1 == direction:
                    self.edge_to_set_of_output_nodes_dict[edge].add(node)
                    self.node_to_set_of_output_edges_dict[node].add(edge)

                if 0 == direction:
                    self.edge_to_set_of_regulation_nodes_dict[edge].add(node)
                    self.node_to_set_of_regulation_edges_dict[node].add(edge)

        def update_delete(self, arg: Relationship):
            node = arg.node
            edge = arg.edge
            direction = arg.direction

            self.edge_to_set_of_nodes_dict[edge].remove(node)
            self.node_to_set_of_edges_dict[node].remove(edge)

            if -1 == direction:
                self.edge_to_set_of_input_nodes_dict[edge].remove(node)
                self.node_to_set_of_input_edges_dict[node].remove(edge)

            if 1 == direction:
                self.edge_to_set_of_output_nodes_dict[edge].remove(node)
                self.node_to_set_of_output_edges_dict[node].remove(edge)

            if 0 == direction:
                self.edge_to_set_of_regulation_nodes_dict[edge].remove(node)
                self.node_to_set_of_regulation_edges_dict[node].remove(edge)

    def __init__(self):
        self.pair_of_node_and_component_obs = self.__PairOfNodeAndComponentObs()
        self.relationship_obs = self.__RelationshipObs()


    def information_dict(self):
        num_of_attributes: int = len(self.pair_of_node_and_component_obs.attribute_to_set_of_nodes_dict.keys())
        num_of_nodes: int = len(self.pair_of_node_and_component_obs.node_to_set_of_attributes_dict.keys())
        num_of_edges: int = len(self.relationship_obs.edge_to_set_of_nodes_dict.keys())

        num_of_pair_of_entity_and_component: int = 0
        for node, set_of_attributes in self.pair_of_node_and_component_obs.node_to_set_of_attributes_dict.items():
            num_of_pair_of_entity_and_component += len(set_of_attributes)

        num_of_relationships: int = 0
        for edge, set_of_nodes in self.relationship_obs.edge_to_set_of_nodes_dict.items():
            num_of_relationships += len(set_of_nodes)

        num_of_input_relationships: int = 0
        for edge, set_of_input_nodes in self.relationship_obs.edge_to_set_of_input_nodes_dict.items():
            num_of_input_relationships += len(set_of_input_nodes)

        num_of_output_relationships: int = 0
        for edge, set_of_output_nodes in self.relationship_obs.edge_to_set_of_output_nodes_dict.items():
            num_of_output_relationships += len(set_of_output_nodes)

        num_of_regulation_relationships: int = 0
        for edge, set_of_regulation_nodes in self.relationship_obs.edge_to_set_of_regulation_nodes_dict.items():
            num_of_regulation_relationships += len(set_of_regulation_nodes)

        edge_with_dif_num_relationships: list[int] = list()

        edge_with_dif_num_input_relationships_count: list[int] = list()

        edge_with_dif_num_output_relationships_count: list[int] = list()

        edge_with_dif_num_regulation_relationships_count: list[int] = list()


