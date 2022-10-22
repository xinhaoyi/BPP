from __future__ import annotations
import time

import numpy as np
from file_processor import FileProcessor
from extract_pathway import ReactomeProcessor



class DataDivider:
    def __init__(self):
        self.file_processor = FileProcessor()
        self.edges_file_name = "edges.txt"
        self.nodes_file_name = "nodes.txt"
        self.relationship_file_name = "relationship.txt"
        self.all_components_file_name = "components-all.txt"
        self.entities_components_mapping_file_name = "components-mapping.txt"

        self.entity_id_index_of_relationship = 0
        self.reaction_id_index_of_relationship = 1
        self.direction_index_of_relationship = 2

    def read_reactions_of_one_pathway_from_file(self, pathway_name):
        reactions_ids = self.file_processor.read_file_via_lines("data/" + pathway_name + "/", self.edges_file_name)
        return reactions_ids

    def read_entities_of_one_pathway_from_file(self, pathway_name):
        entities_ids = self.file_processor.read_file_via_lines("data/" + pathway_name + "/", self.nodes_file_name)
        return entities_ids

    def read_relationship_between_reaction_and_nodes_of_one_pathway_from_file(self, pathway_name):
        relationships = self.file_processor.read_file_via_lines("data/" + pathway_name + "/",
                                                                self.relationship_file_name)

        reaction_to_relationship_list_dic: dict[str, list] = {}
        entity_to_relationship_list_dic: dict[str, list] = {}
        for relationship in relationships:
            # 13,192,-1.0
            # entity_id_index, reaction_id_index, direction
            elements = relationship.split(",")
            reaction_id = elements[self.reaction_id_index_of_relationship]
            entity_id = elements[self.entity_id_index_of_relationship]

            if reaction_id in reaction_to_relationship_list_dic.keys():
                relationship_list = reaction_to_relationship_list_dic[reaction_id]
                relationship_list.append(relationship)
            else:
                relationship_list = list()
                relationship_list.append(relationship)
                reaction_to_relationship_list_dic[reaction_id] = relationship_list

            if entity_id in entity_to_relationship_list_dic.keys():
                relationship_list = entity_to_relationship_list_dic[entity_id]
                relationship_list.append(relationship)
            else:
                relationship_list = list()
                relationship_list.append(relationship)
                entity_to_relationship_list_dic[entity_id] = relationship_list

        return relationships, reaction_to_relationship_list_dic, entity_to_relationship_list_dic

    def read_all_components_of_one_pathway_from_file(self, pathway_name):
        components = self.file_processor.read_file_via_lines("data/" + pathway_name + "/", self.all_components_file_name)
        return components


    def read_entities_components_mappings_of_one_pathway_from_files(self, pathway_name):
        entities_components_mappings = self.file_processor.read_file_via_lines("data/" + pathway_name + "/", self.entities_components_mapping_file_name)
        return entities_components_mappings



    def get_divided_reactions_of_pathway(self, pathway_name):
        reactions_ids = self.read_reactions_of_one_pathway_from_file(pathway_name)
        np.random.shuffle(reactions_ids)
        # 7 : 2 : 1
        # train, validation, test
        total_num = len(reactions_ids)
        train_num = int(total_num * 0.7)
        validation_num = int(total_num * 0.2)
        test_num = total_num - train_num - validation_num

        validation_start_index = train_end_index = train_num

        test_start_index = validation_end_index = train_num + validation_num

        train_reactions = reactions_ids[0:train_end_index]
        validation_reactions = reactions_ids[validation_start_index:validation_end_index]
        test_reactions = reactions_ids[test_start_index:]

        return train_reactions, validation_reactions, test_reactions

    def __get_divided_entities_based_on_divided_reactions_of_pathway(self, pathway_name, divided_reactions):
        all_entyities_ids = self.read_entities_of_one_pathway_from_file(pathway_name)
        all_reactions_ids = self.read_reactions_of_one_pathway_from_file(pathway_name)

        relationships, reaction_to_relationship_list_dic, entity_to_relationship_list_dic = self.read_relationship_between_reaction_and_nodes_of_one_pathway_from_file(
            pathway_name)

        reaction_id_mapping_index_dic = {reaction_id: reaction_index for reaction_index, reaction_id in enumerate(all_reactions_ids)}

        divided_entities_set = set()

        for reaction_id in divided_reactions:
            reaction_index = str(reaction_id_mapping_index_dic[reaction_id])
            # reaction_index -> {entity_index,reaction_index,direction}
            if reaction_index in reaction_to_relationship_list_dic.keys():
                relationship_list = reaction_to_relationship_list_dic[reaction_index]
                for relationship in relationship_list:
                    elements = relationship.split(",")
                    entity_index = elements[self.entity_id_index_of_relationship]
                    entity_id = all_entyities_ids[int(entity_index)]
                    divided_entities_set.add(entity_id)
            else:
                print('error! we can\'t find ' + reaction_id)

            # relationship = reaction_relationships_dic[reaction_index]
            # elements = relationship.split(",")
            # entity_index = elements[self.entity_id_index_of_relationship]
            # entity_id = all_entyities_ids[int(entity_index)]
            # divided_entities_set.add(entity_id)

        divided_entities_list = list(divided_entities_set)
        return divided_entities_list

    def get_divided_entities_of_pathway(self, pathway_name, train_reactions, validation_reactions, test_reactions):
        train_entities = self.__get_divided_entities_based_on_divided_reactions_of_pathway(pathway_name, train_reactions)
        validation_entities = self.__get_divided_entities_based_on_divided_reactions_of_pathway(pathway_name, validation_reactions)
        test_entities = self.__get_divided_entities_based_on_divided_reactions_of_pathway(pathway_name, test_reactions)

        return train_entities, validation_entities, test_entities


    def __get_divided_relationship_based_on_divided_reactions_and_divided_entities_of_pathway(self, pathway_name, divided_reactions, divided_entities):
        all_entyities_ids = self.read_entities_of_one_pathway_from_file(pathway_name)
        all_reactions_ids = self.read_reactions_of_one_pathway_from_file(pathway_name)
        relationships, reaction_to_relationship_list_dic, entity_to_relationship_list_dic = self.read_relationship_between_reaction_and_nodes_of_one_pathway_from_file(
            pathway_name)
        divided_relationships = []
        # new_index_for_entity = 0

        reaction_id_mapping_all_reactions_index_dic = {reaction_id: reaction_index for reaction_index, reaction_id in
                                         enumerate(all_reactions_ids)}

        reaction_id_mapping_divided_reactions_index_dic = {reaction_id: reaction_index for reaction_index, reaction_id
                                                           in enumerate(divided_reactions)}

        entity_id_mapping_divided_entities_index_dic = {entity_id: entity_index for entity_index, entity_id
                                                           in enumerate(divided_entities)}

        for reaction_id in divided_reactions:
            reaction_index = str(reaction_id_mapping_all_reactions_index_dic[reaction_id])
            if reaction_index in reaction_to_relationship_list_dic.keys():
                relationship_list = reaction_to_relationship_list_dic[reaction_index]

                for relationship in relationship_list:
                    elements = relationship.split(",")
                    entity_index = elements[self.entity_id_index_of_relationship]
                    entity_id = all_entyities_ids[int(entity_index)]

                    direction = elements[self.direction_index_of_relationship]

                    new_reaction_index = reaction_id_mapping_divided_reactions_index_dic[reaction_id]

                    new_entity_index = entity_id_mapping_divided_entities_index_dic[entity_id]

                    line_message = str(new_entity_index) + "," + str(new_reaction_index) + "," + direction

                    divided_relationships.append(line_message)

        return divided_relationships


    def get_divided_relationship_of_pathway(self, pathway_name, train_reactions, validation_reactions, test_reactions, train_entities, validation_entities, test_entities):
        train_relationships = self.__get_divided_relationship_based_on_divided_reactions_and_divided_entities_of_pathway(pathway_name, train_reactions, train_entities)
        validation_relationships = self.__get_divided_relationship_based_on_divided_reactions_and_divided_entities_of_pathway(pathway_name, validation_reactions, validation_entities)
        test_relationships = self.__get_divided_relationship_based_on_divided_reactions_and_divided_entities_of_pathway(
            pathway_name, test_reactions, test_entities)

        return train_relationships, validation_relationships, test_relationships


    def __get_divided_components_based_on_divided_entities_of_pathway(self, pathway_name, divided_entities):
        all_components = self.read_all_components_of_one_pathway_from_file(pathway_name)
        all_entities = self.read_entities_of_one_pathway_from_file(pathway_name)

        all_component_id_to_component_index_dic = {component_id: component_index for component_index, component_id in enumerate(all_components)}
        all_entities_id_to_entities_index_dic = {entities_id: entities_index for entities_index, entities_id in enumerate(all_entities)}


        # index_of_componentsA,index_of_componentsB,index_of_componentsC
        entities_components_mappings = self.read_entities_components_mappings_of_one_pathway_from_files(pathway_name)

        # components_set_of_pathway = set()
        components_set_of_pathway = []

        for entity_id in divided_entities:
            entity_index = all_entities_id_to_entities_index_dic[entity_id]
            components_of_single_entity = entities_components_mappings[entity_index]
            components_index_of_single_entity = components_of_single_entity.split(",")

            for component_index in components_index_of_single_entity:
                component_id = all_components[int(component_index)]
                # components_set_of_pathway.add(component_id)
                components_set_of_pathway.append(component_id)

        components_list_of_pathway = list(components_set_of_pathway)
        return components_list_of_pathway

    def get_divided_components_of_pathway(self, pathway_name, train_entities, validation_entities, test_entities):
        train_components = self.__get_divided_components_based_on_divided_entities_of_pathway(pathway_name, train_entities)
        validation_components = self.__get_divided_components_based_on_divided_entities_of_pathway(pathway_name, validation_entities)
        test_components = self.__get_divided_components_based_on_divided_entities_of_pathway(pathway_name, test_entities)

        return train_components, validation_components, test_components

    def __get_divided_components_mapping_based_on_entities_of_pathway(self, pathway_name, divided_entities):
        all_components = self.read_all_components_of_one_pathway_from_file(pathway_name)
        all_entities = self.read_entities_of_one_pathway_from_file(pathway_name)

        all_component_id_to_component_index_dic = {component_id: component_index for component_index, component_id in
                                                   enumerate(all_components)}
        all_entities_id_to_entities_index_dic = {entities_id: entities_index for entities_index, entities_id in
                                                 enumerate(all_entities)}

        # index_of_componentsA,index_of_componentsB,index_of_componentsC
        entities_to_components_mappings = self.read_entities_components_mappings_of_one_pathway_from_files(pathway_name)

        # components_set_of_pathway = set()
        components_mapping_list = []

        for entity_id in divided_entities:
            entity_index = all_entities_id_to_entities_index_dic[entity_id]
            components_of_single_entity = entities_to_components_mappings[entity_index]
            components_mapping_list.append(components_of_single_entity)

        return components_mapping_list


    def get_divided_components_mapping_of_pathway(self, pathway_name, train_entities, validation_entities, test_entities):

        train_components_mapping = self.__get_divided_components_mapping_based_on_entities_of_pathway(pathway_name, train_entities)
        validation_components_mapping = self.__get_divided_components_mapping_based_on_entities_of_pathway(pathway_name, validation_entities)
        test_components_mapping = self.__get_divided_components_mapping_based_on_entities_of_pathway(pathway_name, test_entities)

        return train_components_mapping, validation_components_mapping, test_components_mapping



    def execute_divide_dataset_of_pathway_randomly(self, pathway_name):
        train_reactions, validation_reactions, test_reactions = self.get_divided_reactions_of_pathway(
            pathway_name)
        pre_path = "data/" + pathway_name + "/divided_dataset_methodA/"
        self.file_processor.createFile(pre_path + "train", self.edges_file_name)
        self.file_processor.createFile(pre_path + "validation", self.edges_file_name)
        self.file_processor.createFile(pre_path + "test", self.edges_file_name)

        self.file_processor.writeMessageToFile(pre_path + "train", self.edges_file_name,
                                               train_reactions)
        self.file_processor.writeMessageToFile(pre_path + "validation",
                                               self.edges_file_name, validation_reactions)
        self.file_processor.writeMessageToFile(pre_path + "test", self.edges_file_name,
                                               test_reactions)

        train_entities, validation_entities, test_entities = self.get_divided_entities_of_pathway(pathway_name, train_reactions, validation_reactions, test_reactions)
        self.file_processor.createFile(pre_path + "train", self.nodes_file_name)
        self.file_processor.createFile(pre_path + "validation", self.nodes_file_name)
        self.file_processor.createFile(pre_path + "test", self.nodes_file_name)

        self.file_processor.writeMessageToFile(pre_path + "train", self.nodes_file_name,
                                               train_entities)
        self.file_processor.writeMessageToFile(pre_path + "validation",
                                               self.nodes_file_name, validation_entities)
        self.file_processor.writeMessageToFile(pre_path + "test", self.nodes_file_name,
                                               test_entities)

        train_relationships, validation_relationships, test_relationships = self.get_divided_relationship_of_pathway(pathway_name, train_reactions, validation_reactions, test_reactions, train_entities, validation_entities, test_entities)

        self.file_processor.createFile(pre_path + "train", self.relationship_file_name)
        self.file_processor.createFile(pre_path + "validation", self.relationship_file_name)
        self.file_processor.createFile(pre_path + "test", self.relationship_file_name)

        self.file_processor.writeMessageToFile(pre_path + "train", self.relationship_file_name,
                                               train_relationships)
        self.file_processor.writeMessageToFile(pre_path + "validation",
                                               self.relationship_file_name, validation_relationships)
        self.file_processor.writeMessageToFile(pre_path + "test", self.relationship_file_name,
                                               test_relationships)

        # train_components, validation_components, test_components = self.get_divided_components_of_pathway(pathway_name, train_entities, validation_entities, test_entities)

        # self.file_processor.delete_file(pre_path + "train", self.all_components_file_name)
        # self.file_processor.delete_file(pre_path + "validation", self.all_components_file_name)
        # self.file_processor.delete_file(pre_path + "test", self.all_components_file_name)

        # self.file_processor.createFile(pre_path + "train", self.all_components_file_name)
        # self.file_processor.createFile(pre_path + "validation", self.all_components_file_name)
        # self.file_processor.createFile(pre_path + "test", self.all_components_file_name)
        #
        # self.file_processor.writeMessageToFile(pre_path + "train", self.all_components_file_name,
        #                                        train_components)
        # self.file_processor.writeMessageToFile(pre_path + "validation",
        #                                        self.all_components_file_name, validation_components)
        # self.file_processor.writeMessageToFile(pre_path + "test", self.all_components_file_name,
        #                                        test_components)

        train_components_mapping, validation_components_mapping, test_components_mapping = self.get_divided_components_mapping_of_pathway(pathway_name, train_entities, validation_entities, test_entities)

        self.file_processor.createFile(pre_path + "train", self.entities_components_mapping_file_name)
        self.file_processor.createFile(pre_path + "validation", self.entities_components_mapping_file_name)
        self.file_processor.createFile(pre_path + "test", self.entities_components_mapping_file_name)

        self.file_processor.writeMessageToFile(pre_path + "train", self.entities_components_mapping_file_name,
                                               train_components_mapping)
        self.file_processor.writeMessageToFile(pre_path + "validation",
                                               self.entities_components_mapping_file_name, validation_components_mapping)
        self.file_processor.writeMessageToFile(pre_path + "test", self.entities_components_mapping_file_name,
                                               test_components_mapping)




if __name__ == '__main__':

    time_start = time.time()  # record the start time

    data_divider = DataDivider()

    reactome_processor = ReactomeProcessor('neo4j', '123456')

    top_pathways = reactome_processor.get_all_top_pathways()

    for pathway_id in top_pathways:
        pathway_name = reactome_processor.get_pathway_name_by_id(pathway_id)
        data_divider.execute_divide_dataset_of_pathway_randomly(pathway_name)
    # data_divider.execute_divide_dataset_of_pathway_randomly("Autophagy")

    time_end = time.time()  # record the ending time

    time_sum = time_end - time_start  # The difference is the execution time of the program in seconds

    print("success! it takes " + str(time_sum) + " seconds to divide the dataset")
