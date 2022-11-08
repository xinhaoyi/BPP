import re

from file_processor import FileProcessor
from property import Properties


class DataBean:
    def __init__(self, pathway: str, is_raw_dataset: bool = True, is_divided_dataset: bool = False,
                 is_combination_dataset: bool = True, divided_dataset_task: str = None,
                 divided_dataset_type: str = None):

        self.__entity_id_index_of_relationship = 0
        self.__reaction_id_index_of_relationship = 1
        self.__direction_index_of_relationship = 2

        self.__file_processor = FileProcessor()
        self.__file_name_properties = Properties("file_name.properties")
        self.__pathway_name_path_properties = Properties("pathway_name_path.properties")

        self.__reactions: list[str] = list()
        self.__entities: list[str] = list()
        self.__components: list[str] = list()
        self.__relationships: list[list[str]] = list()
        self.__list_of_pair_of_entity_and_component: list[list[str]] = list()

        self.is_raw_dataset = is_raw_dataset
        self.is_divided_dataset = is_divided_dataset
        self.is_combination_dataset = is_combination_dataset

        self.__divided_dataset_task = divided_dataset_task
        self.__divided_dataset_type = divided_dataset_type

        self.__pathway = re.sub(r"\s+", '_', pathway)
        self.__path = ""

        self.__check_data_bean_status()

        self.__init_path()

    def __check_data_bean_status(self):
        pathway_name_allowance_list = self.__file_processor.read_file_via_lines()

        if self.is_raw_dataset and self.is_divided_dataset:
            raise Exception(
                "Status Conflict! This Data Bean can't be a raw dataset and a divided dataset at the same time")

        if self.is_raw_dataset and self.is_combination_dataset:
            raise Exception(
                "Status Conflict! This Data Bean can't be a raw dataset and a combination dataset at the same time")

        divided_dataset_task_allowance_list = ["attribute prediction dataset", "input link prediction dataset",
                                               "output link prediction dataset"]
        divided_dataset_type_allowance_list = ["test", "train", "validation"]

        if None is not self.__divided_dataset_task and self.__divided_dataset_task not in divided_dataset_task_allowance_list:
            raise Exception(
                "Your divided dataset task is illegal, the allowed input is one of \"attribute prediction dataset\", \"input link prediction dataset\", \"output link prediction dataset\"")

        if None is not self.__divided_dataset_type and self.__divided_dataset_type not in divided_dataset_type_allowance_list:
            raise Exception(
                "Your divided dataset task is illegal, the allowed input is one of \"test\", \"train\", \"validation\"")

    def __init_path(self):
        pathway_with_no_space = re.sub(r"\s+", '_', self.__pathway)
        self.__path = self.__pathway_name_path_properties.get(pathway_with_no_space)

        # not found the pathway and its corresponding path, it's defined as a combination one
        if "" == self.__path and True is self.is_combination_dataset:
            self.__path = "data/" + self.__pathway + "/"

        if None is not self.__divided_dataset_task and None is not self.__divided_dataset_type:
            self.__path = self.__path + self.__divided_dataset_task + "/" + self.__divided_dataset_type + "/"

    def init_inner_elements_from_file(self):
        self.__generate_inner_reactions_from_file()
        self.__generate_inner_entities_from_file()
        self.__generate_inner_components_from_file()
        self.__generate_inner_relationships_from_file()
        self.__generate_inner_pair_of_entity_and_component_from_file()


    def __generate_inner_reactions_from_file(self) -> None:
        reactions_ids_file_name = self.__file_name_properties.get("reactions_ids_file_name")
        self.__reactions = self.__file_processor.read_file_via_lines(self.__path, reactions_ids_file_name)

    # data/Neuronal System/divided_dataset_methodA/test/components-mapping.txt
    def __generate_inner_entities_from_file(self) -> None:
        entities_ids_file_name = self.__file_name_properties.get("entities_ids_file_name")
        self.__entities = self.__file_processor.read_file_via_lines(self.__path, entities_ids_file_name)

    #
    def __generate_inner_components_from_file(self) -> None:
        components_ids_file_name = self.__file_name_properties.get("components_ids_file_name")
        self.__components = self.__file_processor.read_file_via_lines(self.__path, components_ids_file_name)

    def __generate_inner_relationships_from_file(self) -> None:
        relationships_ids_file_name = self.__file_name_properties.get("relationships_ids_file_name")

        relationships_string_style = self.__file_processor.read_file_via_lines(self.__path,
                                                                               relationships_ids_file_name)

        for relationship in relationships_string_style:
            # 13,192,-1.0
            # entity_id_index, reaction_id_index, direction
            # line_of_reaction_id_and_entity_id_and_direction
            elements = relationship.split(",")

            entity_index = elements[self.__entity_id_index_of_relationship]
            entity_id = self.__entities[int(entity_index)]

            reaction_index = elements[self.__reaction_id_index_of_relationship]
            reaction_id = self.__reactions[int(reaction_index)]

            direction = elements[self.__direction_index_of_relationship]

            line_of_reaction_id_and_entity_id_and_direction: list[str] = list()

            line_of_reaction_id_and_entity_id_and_direction.append(entity_id)
            line_of_reaction_id_and_entity_id_and_direction.append(reaction_id)
            line_of_reaction_id_and_entity_id_and_direction.append(direction)

            # entity_id_index, reaction_id_index, direction
            self.__relationships.append(line_of_reaction_id_and_entity_id_and_direction)

        # list of [entity_id, component_id]

    def __generate_inner_pair_of_entity_and_component_from_file(self) -> None:

        entities_components_association_file_name = self.__file_name_properties.get(
            "entities_components_association_file_name")

        # 355,1190,1209
        list_of_entity_components_mappings_with_index_style = self.__file_processor.read_file_via_lines(
            self.__path, entities_components_association_file_name)

        for i in range(len(list_of_entity_components_mappings_with_index_style)):
            entity_id = self.__entities[i]
            components_str = list_of_entity_components_mappings_with_index_style[i]
            list_of_component_index_str_style = components_str.split(",")

            for component_str in list_of_component_index_str_style:
                line_list_of_entity_id_and_component_id: list[str] = list()
                component_index = int(component_str)
                component_id = self.__components[component_index]
                line_list_of_entity_id_and_component_id.append(entity_id)
                line_list_of_entity_id_and_component_id.append(component_id)
                self.__list_of_pair_of_entity_and_component.append(line_list_of_entity_id_and_component_id)
