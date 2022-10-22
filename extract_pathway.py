# from neo4j import GraphDatabase
import numpy as np
# from py2neo import Graph, Node, Relationship
from py2neo import Graph
import os
import time
from pyecharts import options as opts
from pyecharts.charts import Bar


# The processor which deals with the pathway
class PathWayProcessor:
    '''

    '''
    def __init__(self, graph: Graph):
        self.__graph = graph

    # get_all_top_level_pathways(self):
    # output:
    def get_all_top_level_pathways(self):
        # TopLevelPathway
        # "MATCH (n:TopLevelPathway) WHERE n.speciesName='Homo sapiens' RETURN n.schemaClass"
        # "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway', 'Pathway']) AND n.speciesName='Homo sapiens' RETURN n LIMIT 20"
        # "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway']) AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName"
        __instruction__ = "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway']) AND n.speciesName='Homo sapiens' RETURN n.stId"
        toplevel_pathways = self.__graph.run(__instruction__).to_ndarray()

        # Here, we means reducing output to one dimension
        toplevel_pathways = toplevel_pathways.flatten(order='C').tolist()

        return toplevel_pathways


# The processor which deals with the reaction
class ReactionProcessor:
    def __init__(self, graph: Graph):
        self.__graph = graph
        # self.id_index = 0
        # self.name_index = 1
        self.entity_id_index = 0
        self.reaction_id_index = 1
        self.direction_index = 2

    def get_reactions_from_pathway(self, pathway_stId):
        # we used to return "reaction_stId, reaction_displayName", but now only "reaction_stId"
        __instruction__ = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = '" + str(
            pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        reactions = self.__graph.run(__instruction__).to_ndarray()
        reactions = reactions.flatten(order='C').tolist()
        # There'll be some duplicated reactions via our method in one pathway, so just reduce the duplicated ones
        reactions = list(set(reactions))

        # list of reaction_id
        return reactions

    def get_all_reactions_of_homo_sapiens_in_Reactome(self) -> list:
        reactions = self.__graph.run(
            "MATCH (n:Reaction) WHERE n.speciesName='Homo sapiens' RETURN n.stId").to_ndarray()
        reactions = reactions.flatten(order='C').tolist()
        return reactions

    def get_all_reactions_of_homo_sapiens_based_on_all_top_pathways_in_Reactome(self, top_pathways) -> list:
        reactions_set = set()
        for top_pathway_id in top_pathways:
            reactions = self.get_reactions_from_pathway(top_pathway_id)
            for reaction in reactions:
                reactions_set.add(reaction)
        return list(reactions_set)

    def __get_input_relationship_edge_of_reaction(self, reaction_stId):
        __instruction_input__ = "MATCH (n:Reaction)-[r:input]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, n.stId"
        __input_edges__ = self.__graph.run(__instruction_input__).to_ndarray()

        # get the line number of ndarray，which represents the number of input
        num_of_input_edges = __input_edges__.shape[0]

        # we build a complementary vertex based on the number of input
        supplement_vector = np.negative(np.ones(num_of_input_edges))

        # we add a new column, then fill it with 0, which represents input
        __input_edges__ = np.insert(__input_edges__, 2, supplement_vector, axis=1)

        # PhysicalEntity_id, Reaction_id, 0
        return __input_edges__.tolist()

    def __get_output_relationship_edge_of_reaction(self, reaction_stId):
        __instruction_output__ = "MATCH (n:Reaction)-[r:output]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, n.stId"
        __output_edges__ = self.__graph.run(__instruction_output__).to_ndarray()

        # get the line number of ndarray，which represents the number of input
        __num_of_output_edges = __output_edges__.shape[0]

        # we build a complementary vertex based on the number of input
        supplement_vector = np.ones(__num_of_output_edges)

        # we add a new column, then fill it with 0, which represents input
        __output_edges__ = np.insert(__output_edges__, 2, supplement_vector, axis=1)

        # PhysicalEntity_id, Reaction_id, 1
        return __output_edges__.tolist()

    def get_all_related_edges_of_single_reaction(self, reaction_stId):
        # R-HSA-9613352
        # R-HSA-9646383
        input_edges__ = self.__get_input_relationship_edge_of_reaction(reaction_stId)
        output_edges__ = self.__get_output_relationship_edge_of_reaction(reaction_stId)
        edges_for_reaction = np.vstack((input_edges__, output_edges__))

        # PhysicalEntity_id, Reaction_id, 0/1
        return edges_for_reaction.tolist()

    def get_all_unique_edges_of_set_of_reactions(self, reaction_stIds):
        edges_for_set_of_reactions = np.empty(shape=(0, 3))
        for reaction_stId in reaction_stIds:
            edges_for_single_reaction = self.get_all_related_edges_of_single_reaction(reaction_stId)
            edges_for_set_of_reactions = np.vstack((edges_for_set_of_reactions, edges_for_single_reaction))

        # reduce the duplicate ones
        # PhysicalEntity_id, Reaction_id, 0/1    -a list
        unique_edges_for_set_of_reactions = list(set(tuple(edge) for edge in edges_for_set_of_reactions))

        # every edge from tuple to list
        unique_edges_for_set_of_reactions = list(list(edge) for edge in unique_edges_for_set_of_reactions)

        return unique_edges_for_set_of_reactions

    def get_physical_entities_from_single_reaction(self, reaction_stId):
        __instruction_input__ = "MATCH (n:Reaction)-[r:input]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        input_entities = self.__graph.run(__instruction_input__).to_ndarray().flatten(order='C').tolist()

        __instruction_output__ = "MATCH (n:Reaction)-[r:output]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        output_entities = self.__graph.run(__instruction_output__).to_ndarray().flatten(order='C').tolist()

        physical_entities = input_entities + output_entities

        # list of PhysicalEntity_id
        return physical_entities

    '''
        __get_physical_entities_from_set_of_reactions(self, reactions):
        input: reactions - list of reaction_id
        output: physical entities - list of physical_entity_id
    '''

    def get_unique_physical_entities_from_set_of_reactions(self, reactions):
        physical_entities_set = set()

        for reaction_stId in reactions:
            # list of PhysicalEntity_id
            physical_entities = self.get_physical_entities_from_single_reaction(reaction_stId)

            for physical_entity in physical_entities:
                physical_entities_set.add(physical_entity)

        physical_entities = list(physical_entities_set)

        return physical_entities


# The processor which deals with the physical entity
class PhysicalEntityProcessor:
    # TypeDetectorOfPhysicalEntity is the inner class to help to define a specific type for the physical entity
    class TypeDetectorOfPhysicalEntity:
        complex_arr = ['Complex']
        polymer_arr = ['Polymer']
        genomeEncodedEntity_arr = ['GenomeEncodedEntity', 'EntityWithAccessionedSequence']
        entitySet_arr = ['EntitySet', 'CandidateSet', 'DefinedSet']
        simpleEntity_arr = ['SimpleEntity']
        otherEntity_arr = ['OtherEntity']
        drug_arr = ['Drug', 'ChemicalDrug', 'ProteinDrug']
        type_dic = dict(complex_type=complex_arr, polymer_type=polymer_arr,
                        genomeEncodedEntity_type=genomeEncodedEntity_arr, entitySet_type=entitySet_arr,
                        simpleEntity_type=simpleEntity_arr, otherEntity_type=otherEntity_arr, drug_type=drug_arr)

        def __init__(self):
            pass

        def __is_complex(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('complex_type')

        def __is_polymer(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('polymer_type')

        def __is_genomeEncodedEntity(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('genomeEncodedEntity_type')

        def __is_entitySet(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('entitySet_type')

        def __is_simpleEntity(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('simpleEntity_type')

        def __is_otherEntity(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('otherEntity_type')

        def __is_drug(self, input_entity_schemaClass):
            return str(input_entity_schemaClass) in self.type_dic.get('drug_type')

        def get_type_of_physical_entity(self, input_entity_schemaClass):
            if (self.__is_complex(input_entity_schemaClass)
                    or self.__is_polymer(input_entity_schemaClass)
                    or self.__is_simpleEntity(input_entity_schemaClass)
                    or self.__is_otherEntity(input_entity_schemaClass)):
                # 'Complex' 'Polymer' 'SimpleEntity' 'OtherEntity'
                return input_entity_schemaClass
            elif (self.__is_genomeEncodedEntity(input_entity_schemaClass)):
                return 'GenomeEncodedEntity'
            elif (self.__is_entitySet(input_entity_schemaClass)):
                return 'EntitySet'
            elif (self.__is_drug(input_entity_schemaClass)):
                return 'Drug'
        # end of the inner class -> TypeDetectorOfPhysicalEntity

    # the __init__ for the outer class "PhysicalEntityProcessor"
    def __init__(self, graph: Graph):
        self.__graph = graph
        self.__type_detector = self.TypeDetectorOfPhysicalEntity()
        self.__funcs_for_get_components = {'Complex': self.__get_components_of_complex,
                                           'Polymer': self.__get_components_of_polymer,
                                           'GenomeEncodedEntity': self.__get_components_of_GenomeEncodedEntity,
                                           'EntitySet': self.__get_components_of_EntitySet,
                                           'SimpleEntity': self.__get_components_of_SimpleEntity,
                                           'OtherEntity': self.__get_components_of_OtherEntity,
                                           'Drug': self.__get_components_of_Drug}
        self.id_index = 0
        self.name_index = 1

    def __get_components_of_complex(self, physical_entity_id: str):
        # MATCH (n:Complex)-[r:hasComponent]->(m) WHERE n.stId = 'R-HSA-917704' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:Complex)-[r:hasComponent]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        # if Complex has no components, we define itself as a component.
        if (len(__components__) == 0):
            # MATCH (n:Complex) WHERE n.stId = 'R-HSA-917704' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:Complex) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId"
            __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        return __components__

    def __get_components_of_polymer(self, physical_entity_id: str):
        # MATCH (n:Polymer)-[r:repeatedUnit]->(m) WHERE n.stId = 'R-HSA-9626247' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:Polymer)-[r:repeatedUnit]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        # if Polymer has no components, we define itself as a component.
        if (len(__components__) == 0):
            # MATCH (n:Polymer) WHERE n.stId = 'R-HSA-2214302' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:Polymer) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId"
            __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        return __components__

    # GenomeEncodedEntity has no components, so define itself as an attribute
    def __get_components_of_GenomeEncodedEntity(self, physical_entity_id: str):
        # MATCH (n:GenomeEncodedEntity) WHERE n.stId = 'R-HSA-2029007' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:GenomeEncodedEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    def __get_components_of_EntitySet(self, physical_entity_id: str):
        # MATCH (n:EntitySet)-[r:hasMember]->(m) WHERE n.stId = 'R-HSA-170079' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:EntitySet)-[r:hasMember]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        # if EntitySet has no members(component), we define itself as a member(component).
        if (len(__components__) == 0):
            # MATCH (n:EntitySet) WHERE n.stId = 'R-HSA-170079' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:EntitySet) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId"
            __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()

        return __components__

    # SimpleEntity has no components, so define itself as an attribute
    # A kind reminder that the SimpleEntity has no n.speciesName attribute
    def __get_components_of_SimpleEntity(self, physical_entity_id: str):
        # MATCH (n:SimpleEntity) WHERE n.stId = 'R-ALL-29438' RETURN n.stId, n.displayName, n.speciesName
        __instruction__ = "MATCH (n:SimpleEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    # OtherEntity has no components, so define itself as an attribute
    # A kind reminder that the OtherEntity has no n.speciesName attribute
    def __get_components_of_OtherEntity(self, physical_entity_id: str):
        # MATCH (n:OtherEntity) WHERE n.stId = 'R-ALL-422139' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:OtherEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    # Drug has no components, so define itself as an attribute
    # A kind reminder that the Drug has no n.speciesName attribute
    def __get_components_of_Drug(self, physical_entity_id: str):
        # MATCH (n:Drug) WHERE n.stId = 'R-ALL-9674322' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:Drug) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId"
        __components__ = self.__graph.run(__instruction__).to_ndarray().flatten(order='C').tolist()
        return __components__

    # get schemaClass of physical entity
    def __get_schemaClass_of_physical_entity(self, physical_entity_id: str):
        __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' AND n.speciesName='Homo sapiens' RETURN n.schemaClass"
        __physical_entity_schemaClass__ndarray = self.__graph.run(__instruction__).to_ndarray()
        # if is None, it's possibly because that the node is An Simple Entity/OtherEntity/Drug which doesn't have n.speciesName
        if __physical_entity_schemaClass__ndarray is None or __physical_entity_schemaClass__ndarray.size == 0:
            __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.schemaClass"
            __physical_entity_schemaClass__ndarray = self.__graph.run(__instruction__).to_ndarray()

        __physical_entity_schemaClass = __physical_entity_schemaClass__ndarray[0, 0]

        return __physical_entity_schemaClass

    def get_components_of_single_physical_entity(self, physical_entity_id: str):
        __physical_entity_schemaClass__ = self.__get_schemaClass_of_physical_entity(physical_entity_id)
        type_of_physical_entity = self.__type_detector.get_type_of_physical_entity(__physical_entity_schemaClass__)
        # func_for_get_components_selected = self.__funcs_for_get_components.get(type_of_physical_entity)
        func_for_get_components_selected = self.__funcs_for_get_components[type_of_physical_entity]
        components = []
        if (func_for_get_components_selected is not None):
            components = func_for_get_components_selected(physical_entity_id)

        # list of physical_entity_id
        return components

    '''
        __get_unique_componets_from_physical_entities(self, physical_entities):
        input: list of physical_entity_id
        output: list of physical_entity_id(components), dict:physical_entity_id(node) -> set(physical_entity_id(component), physical_entity_id(component)....physical_entity_id(component)
        '''

    def get_unique_componets_from_set_of_physical_entities(self, physical_entities):

        component_ids_set = set()

        components_dict = {}

        for physical_entity_id in physical_entities:
            component_ids = self.get_components_of_single_physical_entity(
                physical_entity_id)

            if len(component_ids) == 0:
                print("error! for finding no components -> physical_entity_stId:" + str(physical_entity_id))

            for component_id in component_ids:
                component_ids_set.add(component_id)

            components_dict[physical_entity_id] = set(component_ids)

        component_ids_unique = list(component_ids_set)

        return component_ids_unique, components_dict


# The processor which deal with Reactome DataBase
class ReactomeProcessor:
    def __init__(self, user_name, password):
        self.__link = "bolt://localhost:7687"

        # user_name = 'neo4j'
        self.__user_name = user_name

        # password = '123456'
        self.__password = password

        self.__graph = self.__login(self.__link, self.__user_name, self.__password)

        self.__pathway_processor = PathWayProcessor(self.__graph)

        self.__reaction_processor = ReactionProcessor(self.__graph)

        # PhysicalEntityProcessor
        self.__physical_entity_processor = PhysicalEntityProcessor(self.__graph)

    @staticmethod
    def __login(link, user_name, password):
        graph = Graph(link, auth=(user_name, password))
        return graph

    '''
    get_all_top_pathways(self):
    output: a list of pathway_id
    '''

    def get_all_top_pathways(self):
        toplevel_pathways = self.__pathway_processor.get_all_top_level_pathways()
        print(toplevel_pathways)
        return toplevel_pathways

    def get_pathway_name_by_id(self, pathway_stId):
        __instruction__ = "MATCH (n:Pathway) WHERE n.stId = '" + str(pathway_stId) + "' RETURN n.displayName"
        pathways = self.__graph.run(__instruction__).to_ndarray()
        pathways = pathways.flatten(order='C')
        if (pathways.size == 1):
            return pathways[0]
        else:
            print("sorry, we can't find pathway with stId = '" + str(pathway_stId) + "'")
            return ''

    def get_reaction_name_by_id(self, reaction_stId):
        __instruction__ = "MATCH (n:Reaction) WHERE n.stId = '" + str(reaction_stId) + "' RETURN n.displayName"
        reactions = self.__graph.run(__instruction__).to_ndarray()
        reactions = reactions.flatten(order='C')
        if (reactions.size == 1):
            return reactions[0]
        else:
            print("sorry, we can't find reaction with stId = '" + str(reaction_stId) + "'")
            return ''

    def get_physical_entity_name_by_id(self, physical_entity_stId):
        __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            physical_entity_stId) + "' RETURN n.displayName"
        physical_entities = self.__graph.run(__instruction__).to_ndarray()
        physical_entities = physical_entities.flatten(order='C')
        if (physical_entities.size == 1):
            return physical_entities[0]
        else:
            print("sorry, we can't find physical entity with stId = '" + str(physical_entity_stId) + "'")
            return ''

    def get_all_relationships_for_single_pathway(self, pathway_id):
        reactions = self.__reaction_processor.get_reactions_from_pathway(pathway_id)
        unique_edges_for_single_pathway = self.__reaction_processor.get_all_unique_edges_of_set_of_reactions(reactions)
        return unique_edges_for_single_pathway

    def extract_edges_nodes_relationships_all_components_and_dic_of_entity_components_for_one_pathway(self,
                                                                                                      pathway_stId) -> \
            tuple[list, list, list, list, list]:

        """
        extract
        input: pathway_stId: the id of a pathway
        output: reaction_ids of the pathway, node_ids of the pathway, component_ids of all the nodes of the pathway, dictionary of node_id to a set of its component_ids

        """

        if pathway_stId != -1:
            # normal pathway
            print("\n")
            print("\n")
            print("\033[0;37;41m" + "************" + self.get_pathway_name_by_id(pathway_stId) + "************" + "\033[0m")

            reactions = self.__reaction_processor.get_reactions_from_pathway(pathway_stId)

            # build a dictionary that mapping: reaction_id -> line_index
            reactions_index_dic = {reaction_id: index for index, reaction_id in enumerate(reactions)}

            # get unique physical entities for one pathway
            physical_entity_ids_from_reactions_for_one_pathway = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(
                reactions)

            # build a dictionary that mapping: entity_id -> line_index
            entities_index_dic = {entity_id: index for index, entity_id in
                                  enumerate(physical_entity_ids_from_reactions_for_one_pathway)}

            relationships_between_nodes_edges = self.get_all_relationships_for_single_pathway(pathway_stId)

        else:
            # we'll calculate on the whole reactome

            # The latter instruction is the old version that we will never use,
            # as there will be a small amount(100+) of reactions that belongs to no pathways
            # We just simply quit these datas

            # reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_in_Reactome()

            top_pathways = self.__pathway_processor.get_all_top_level_pathways()

            reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_based_on_all_top_pathways_in_Reactome(
                top_pathways)

            # build a dictionary that mapping: reaction_id -> reaction_index
            reactions_index_dic = {reaction_id: index for index, reaction_id in enumerate(reactions)}

            physical_entity_ids_from_reactions_for_one_pathway = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(
                reactions)

            # build a dictionary that mapping: entity_id -> entity_index
            entities_index_dic = {entity_id: index for index, entity_id in
                                  enumerate(physical_entity_ids_from_reactions_for_one_pathway)}

            relationships_between_nodes_edges = self.__reaction_processor.get_all_unique_edges_of_set_of_reactions(
                reactions)

        relationships_between_nodes_and_edges_with_index_style = []

        # relationship: node_id,reaction_id,direction(-1 or 1)
        for relationship in relationships_between_nodes_edges:
            # node_index,reaction_index,direction(-1 or 1)
            line_message = ""
            entity_id = relationship[self.__reaction_processor.entity_id_index]
            entity_index = entities_index_dic[entity_id]

            reaction_id = relationship[self.__reaction_processor.reaction_id_index]
            reaction_index = reactions_index_dic[reaction_id]

            direaction = relationship[self.__reaction_processor.direction_index]

            line_message = line_message + str(entity_index) + "," + str(reaction_index) + "," + str(direaction)

            relationships_between_nodes_and_edges_with_index_style.append(line_message)

        # remove the duplicate components
        component_ids_unique_for_one_pathway, components_dic = self.__physical_entity_processor.get_unique_componets_from_set_of_physical_entities(
            physical_entity_ids_from_reactions_for_one_pathway)

        # build a dictionary that mapping: component_id -> line_index
        components_index_dic = {component_id: index for index, component_id in
                                enumerate(component_ids_unique_for_one_pathway)}

        # dela with the components message like -
        # a list of ["node_id:component_id,component_id,component_id..", "node_id:component_id,component_id,component_id.."]
        entity_index_to_components_indices_mapping_list = []
        for node_id, set_of_components in components_dic.items():
            node_id_index = entities_index_dic[node_id]
            # component_msg = str(node_id_index) + ":"
            # we won't store the index of entity, as it's in the same sequence with the data in nodes.txt
            component_msg = ""
            list_of_components = []
            for component_id in set_of_components:
                component_id_index = components_index_dic[component_id]
                list_of_components.append(component_id_index)

            list_of_components = sorted(list_of_components)

            for component_id_index in list_of_components:
                component_msg = component_msg + str(component_id_index) + ","

            # remove the comma in the end
            component_msg = component_msg[:-1]
            entity_index_to_components_indices_mapping_list.append(component_msg)

        num_of_edges = str(len(reactions))
        num_of_nodes = str(len(physical_entity_ids_from_reactions_for_one_pathway))
        dimensionality = str(len(component_ids_unique_for_one_pathway))

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        return reactions, physical_entity_ids_from_reactions_for_one_pathway, relationships_between_nodes_and_edges_with_index_style, component_ids_unique_for_one_pathway, entity_index_to_components_indices_mapping_list

    def get_reactions_index_to_list_of_relationships_dic_based_on_relationships(self, relationships: list[str]) -> \
            tuple[dict[str, list], dict[str, list], dict[str, list]]:
        # reactions: a list of reaction_id
        # relationships: a list of {entity_index, reaction_index, direction}
        reaction_index_to_list_of_relationships_dic: dict[str, list] = {}
        reaction_index_to_list_of_input_relationships_dic: dict[str, list] = {}
        reaction_index_to_list_of_output_relationships_dic: dict[str, list] = {}
        for relationship in relationships:
            line_elements = relationship.split(",")
            entity_index = line_elements[self.__reaction_processor.entity_id_index]
            reaction_index = line_elements[self.__reaction_processor.reaction_id_index]
            direction = line_elements[self.__reaction_processor.direction_index]
            if reaction_index in reaction_index_to_list_of_relationships_dic.keys():
                list_of_relationships = reaction_index_to_list_of_relationships_dic[reaction_index]
                list_of_relationships.append(relationship)
            else:
                reaction_index_to_list_of_relationships_dic[reaction_index] = list()
                reaction_index_to_list_of_relationships_dic[reaction_index].append(relationship)

            if float(direction) < 0:
                if reaction_index in reaction_index_to_list_of_input_relationships_dic.keys():
                    list_of_input_relationships = reaction_index_to_list_of_input_relationships_dic[reaction_index]
                    list_of_input_relationships.append(relationship)
                else:
                    reaction_index_to_list_of_input_relationships_dic[reaction_index] = list()
                    reaction_index_to_list_of_input_relationships_dic[reaction_index].append(relationship)

            if float(direction) > 0:
                if reaction_index in reaction_index_to_list_of_output_relationships_dic.keys():
                    list_of_output_relationships = reaction_index_to_list_of_output_relationships_dic[reaction_index]
                    list_of_output_relationships.append(relationship)
                else:
                    reaction_index_to_list_of_output_relationships_dic[reaction_index] = list()
                    reaction_index_to_list_of_output_relationships_dic[reaction_index].append(relationship)

        return reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic

    def get_reaction_status_dic(self, reaction_index_to_list_of_relationships_dic) -> {str: int}:
        reaction_to_relationship_status_dic: {str: int} = {"total_num_of_reactions": 0,
                                                           "num_of_reactions_with_one_relationship": 0,
                                                           "num_of_reactions_with_two_relationships": 0,
                                                           "num_of_reactions_with_three_relationships": 0,
                                                           "num_of_reactions_with_four_relationships": 0,
                                                           "num_of_reactions_with_five_relationships": 0,
                                                           "num_of_reactions_with_six_relationships": 0,
                                                           "num_of_reactions_with_seven_relationships": 0,
                                                           "num_of_reactions_with_eight_relationships": 0,
                                                           "num_of_reactions_with_more_than_eight_relationships": 0}

        dic_key_name: {int: str} = {1: "num_of_reactions_with_one_relationship",
                                    2: "num_of_reactions_with_two_relationships",
                                    3: "num_of_reactions_with_three_relationships",
                                    4: "num_of_reactions_with_four_relationships",
                                    5: "num_of_reactions_with_five_relationships",
                                    6: "num_of_reactions_with_six_relationships",
                                    7: "num_of_reactions_with_seven_relationships",
                                    8: "num_of_reactions_with_eight_relationships"}

        reaction_to_relationship_status_dic["total_num_of_reactions"] = len(reaction_index_to_list_of_relationships_dic)

        for reaction_index, list_of_relationships in reaction_index_to_list_of_relationships_dic.items():
            num_of_relationships = len(list_of_relationships)
            if num_of_relationships in dic_key_name.keys():
                key_name = dic_key_name.get(num_of_relationships)
                temp_val = reaction_to_relationship_status_dic.get(key_name)
                reaction_to_relationship_status_dic[dic_key_name.get(len(list_of_relationships))] = temp_val + 1
            else:
                temp_val = reaction_to_relationship_status_dic.get(
                    "num_of_reactions_with_more_than_eight_relationships")
                reaction_to_relationship_status_dic[
                    "num_of_reactions_with_more_than_eight_relationships"] = temp_val + 1

        return reaction_to_relationship_status_dic

    def print_reaction_status_dic(self, reaction_to_relationship_status_dic: {str: int}, mode: str = ""):
        if "input" == mode:
            mode_message = "input "
        elif "output" == mode:
            mode_message = "output "
        else:
            mode_message = ""

        total_num = reaction_to_relationship_status_dic.get("total_num_of_reactions")
        reaction_num_with_one_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_one_relationship")
        reaction_num_with_two_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_two_relationships")
        reaction_num_with_three_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_three_relationships")
        reaction_num_with_four_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_four_relationships")
        reaction_num_with_five_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_five_relationships")
        reaction_num_with_six_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_six_relationships")
        reaction_num_with_seven_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_seven_relationships")
        reaction_num_with_eight_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_eight_relationships")
        reaction_num_with_more_than_eight_rela = reaction_to_relationship_status_dic.get("num_of_reactions_with_more_than_eight_relationships")

        print("total num of reactions: " + str(total_num))
        print("num of reactions with one " + mode_message + "node: " + str(reaction_num_with_one_rela) + " ( {:.2%}".format(float(reaction_num_with_one_rela) / float(total_num)) + ")")
        print("num of reactions with two " + mode_message + "nodes: " + str(reaction_num_with_two_rela) + " ( {:.2%}".format(float(reaction_num_with_two_rela) / float(total_num)) + ")")
        print("num of reactions with three " + mode_message + "nodes: " + str(reaction_num_with_three_rela) + " ( {:.2%}".format(float(reaction_num_with_three_rela) / float(total_num)) + ")")
        print("num of reactions with four " + mode_message + "nodes: " + str(reaction_num_with_four_rela) + " ( {:.2%}".format(float(reaction_num_with_four_rela) / float(total_num)) + ")")
        print("num of reactions with five " + mode_message + "nodes: " + str(reaction_num_with_five_rela) + " ( {:.2%}".format(float(reaction_num_with_five_rela) / float(total_num)) + ")")
        print("num of reactions with six " + mode_message + "nodes: " + str(reaction_num_with_six_rela) + " ( {:.2%}".format(float(reaction_num_with_six_rela) / float(total_num)) + ")")
        print("num of reactions with seven " + mode_message + "nodes: " + str(reaction_num_with_seven_rela) + " ( {:.2%}".format(float(reaction_num_with_seven_rela) / float(total_num)) + ")")
        print("num of reactions with eight " + mode_message + "nodes: " + str(reaction_num_with_eight_rela) + " ( {:.2%}".format(float(reaction_num_with_eight_rela) / float(total_num)) + ")")
        print("num of reactions with more than eight " + mode_message + "nodes: " + str(reaction_num_with_more_than_eight_rela) + " ( {:.2%}".format(float(reaction_num_with_more_than_eight_rela) / float(total_num)) + ")")

    def get_entity_index_to_list_of_relationships_dic_based_on_relationships(self, relationships: list[str]) -> dict[str, list]:
        entity_index_to_list_of_relationships_dic: dict[str, list] = {}

        for relationship in relationships:
            line_elements = relationship.split(",")
            entity_index = line_elements[self.__reaction_processor.entity_id_index]
            reaction_index = line_elements[self.__reaction_processor.reaction_id_index]
            direction = line_elements[self.__reaction_processor.direction_index]
            if entity_index in entity_index_to_list_of_relationships_dic.keys():
                list_of_relationships = entity_index_to_list_of_relationships_dic[entity_index]
                list_of_relationships.append(relationship)
            else:
                entity_index_to_list_of_relationships_dic[entity_index] = list()
                entity_index_to_list_of_relationships_dic[entity_index].append(relationship)

        return entity_index_to_list_of_relationships_dic

    def get_entity_status_dic(self, entity_index_to_list_of_relationships_dic) -> {str: int}:
        entity_to_relationship_status_dic: {str: int} = {"total_num_of_entities": 0,
                                                           "num_of_entities_with_one_relationship": 0,
                                                           "num_of_entities_with_two_relationships": 0,
                                                           "num_of_entities_with_three_relationships": 0,
                                                           "num_of_entities_with_four_relationships": 0,
                                                           "num_of_entities_with_five_relationships": 0,
                                                           "num_of_entities_with_six_relationships": 0,
                                                           "num_of_entities_with_seven_relationships": 0,
                                                           "num_of_entities_with_eight_relationships": 0,
                                                           "num_of_entities_with_more_than_eight_relationships": 0}

        dic_key_name: {int: str} = {1: "num_of_entities_with_one_relationship",
                                    2: "num_of_entities_with_two_relationships",
                                    3: "num_of_entities_with_three_relationships",
                                    4: "num_of_entities_with_four_relationships",
                                    5: "num_of_entities_with_five_relationships",
                                    6: "num_of_entities_with_six_relationships",
                                    7: "num_of_entities_with_seven_relationships",
                                    8: "num_of_entities_with_eight_relationships"}

        entity_to_relationship_status_dic["total_num_of_entities"] = len(entity_index_to_list_of_relationships_dic)

        for reaction_index, list_of_relationships in entity_index_to_list_of_relationships_dic.items():
            num_of_relationships = len(list_of_relationships)
            if num_of_relationships in dic_key_name.keys():
                key_name = dic_key_name.get(num_of_relationships)
                temp_val = entity_to_relationship_status_dic.get(key_name)
                entity_to_relationship_status_dic[dic_key_name.get(len(list_of_relationships))] = temp_val + 1
            else:
                temp_val = entity_to_relationship_status_dic.get(
                    "num_of_entities_with_more_than_eight_relationships")
                entity_to_relationship_status_dic[
                    "num_of_entities_with_more_than_eight_relationships"] = temp_val + 1

        return entity_to_relationship_status_dic


    def print_entity_status_dic(self, entity_status_dic):
        total_num = entity_status_dic.get("total_num_of_entities")
        entity_num_with_one_rela = entity_status_dic.get("num_of_entities_with_one_relationship")
        entity_num_with_two_rela = entity_status_dic.get("num_of_entities_with_two_relationships")
        entity_num_with_three_rela = entity_status_dic.get(
            "num_of_entities_with_three_relationships")
        entity_num_with_four_rela = entity_status_dic.get(
            "num_of_entities_with_four_relationships")
        entity_num_with_five_rela = entity_status_dic.get(
            "num_of_entities_with_five_relationships")
        entity_num_with_six_rela = entity_status_dic.get("num_of_entities_with_six_relationships")
        entity_num_with_seven_rela = entity_status_dic.get(
            "num_of_entities_with_seven_relationships")
        entity_num_with_eight_rela = entity_status_dic.get(
            "num_of_entities_with_eight_relationships")
        entity_num_with_more_than_eight_rela = entity_status_dic.get(
            "num_of_entities_with_more_than_eight_relationships")

        print("total num of entities: " + str(total_num))
        print("num of entities with one reaction: " + str(
            entity_num_with_one_rela) + " ( {:.2%}".format(
            float(entity_num_with_one_rela) / float(total_num)) + ")")
        print("num of entities with two reactions: " + str(
            entity_num_with_two_rela) + " ( {:.2%}".format(
            float(entity_num_with_two_rela) / float(total_num)) + ")")
        print("num of entities with three reactions: " + str(
            entity_num_with_three_rela) + " ( {:.2%}".format(
            float(entity_num_with_three_rela) / float(total_num)) + ")")
        print("num of entities with four reactions: " + str(
            entity_num_with_four_rela) + " ( {:.2%}".format(
            float(entity_num_with_four_rela) / float(total_num)) + ")")
        print("num of entities with five reactions: " + str(
            entity_num_with_five_rela) + " ( {:.2%}".format(
            float(entity_num_with_five_rela) / float(total_num)) + ")")
        print("num of entities with six reactions: " + str(
            entity_num_with_six_rela) + " ( {:.2%}".format(
            float(entity_num_with_six_rela) / float(total_num)) + ")")
        print("num of entities with seven reactions: " + str(
            entity_num_with_seven_rela) + " ( {:.2%}".format(
            float(entity_num_with_seven_rela) / float(total_num)) + ")")
        print("num of entities with eight reactions: " + str(
            entity_num_with_eight_rela) + " ( {:.2%}".format(
            float(entity_num_with_eight_rela) / float(total_num)) + ")")
        print("num of entities with more than eight reactions: " + str(
            entity_num_with_more_than_eight_rela) + " ( {:.2%}".format(
            float(entity_num_with_more_than_eight_rela) / float(total_num)) + ")")


    def execution_on_single_pathways(self, pathway_stId):
        pathway_name = self.get_pathway_name_by_id(pathway_stId)

        reactions, physical_entity_ids, relationships_between_nodes_edges, component_ids, components_dic = self.extract_edges_nodes_relationships_all_components_and_dic_of_entity_components_for_one_pathway(
            pathway_stId)

        # calculate the data distribution
        reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic = self.get_reactions_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_between_nodes_edges)

        reaction_to_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_relationships_dic)
        reaction_to_input_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_input_relationships_dic)
        reaction_to_output_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_output_relationships_dic)

        print("\033[1;32m" + "For all the relationships:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_relationship_status_dic)
        print("\n")

        print("\033[1;32m" + "For input relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_input_relationship_status_dic, mode="input")
        print("\n")

        print("\033[1;32m" + "For output relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_output_relationship_status_dic, mode="output")
        print("\n")


        entity_index_to_list_of_relationships_dic = self.get_entity_index_to_list_of_relationships_dic_based_on_relationships(relationships_between_nodes_edges)
        entity_to_relationship_status_dic = self.get_entity_status_dic(entity_index_to_list_of_relationships_dic)

        print("\033[1;32m" + "For entities:" + "\033[0m")
        self.print_entity_status_dic(entity_to_relationship_status_dic)
        print("\n")


        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_pathway(pathway_name, reactions, physical_entity_ids,
                                                  relationships_between_nodes_edges, component_ids, components_dic)

        # draw the histogram
        drawer = Drawer(len(reactions), len(physical_entity_ids), len(component_ids), pathway_name)
        drawer.generate_histogram()

    def execution_on_all_pathways(self):
        top_pathways = self.get_all_top_pathways()
        for top_pathway_id in top_pathways:
            self.execution_on_single_pathways(top_pathway_id)

    def execution_on_reactome(self):

        reactions, physical_entities, relationships_with_index_style, components, entity_index_to_components_indices_mapping_list = self.extract_edges_nodes_relationships_all_components_and_dic_of_entity_components_for_one_pathway(
            -1)

        num_of_edges = str(len(reactions))
        num_of_nodes = str(len(physical_entities))
        dimensionality = str(len(components))

        print("************Reactome************")

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        print("For all the relationships:")

        # calculate the data distribution
        reaction_index_to_list_of_relationships_dic, reaction_index_to_list_of_input_relationships_dic, reaction_index_to_list_of_output_relationships_dic = self.get_reactions_index_to_list_of_relationships_dic_based_on_relationships(
            relationships_with_index_style)

        reaction_to_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_relationships_dic)
        reaction_to_input_relationship_status_dic = self.get_reaction_status_dic(reaction_index_to_list_of_input_relationships_dic)
        reaction_to_output_relationship_status_dic = self.get_reaction_status_dic(
            reaction_index_to_list_of_output_relationships_dic)

        print("\033[1;32m" + "For all the relationships:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_relationship_status_dic)
        print("\n")

        print("\033[1;32m" + "For input relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_input_relationship_status_dic, mode="input")
        print("\n")

        print("\033[1;32m" + "For output relationship:" + "\033[0m")
        self.print_reaction_status_dic(reaction_to_output_relationship_status_dic, mode="output")
        print("\n")


        entity_index_to_list_of_relationships_dic = self.get_entity_index_to_list_of_relationships_dic_based_on_relationships(relationships_with_index_style)
        entity_to_relationship_status_dic = self.get_entity_status_dic(entity_index_to_list_of_relationships_dic)

        print("\033[1;32m" + "For entities:" + "\033[0m")
        self.print_entity_status_dic(entity_to_relationship_status_dic)
        print("\n")

        if not os.path.exists("./data/All_data_in_Reactome"):
            os.makedirs("./data/All_data_in_Reactome")

        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_pathway("All_data_in_Reactome", reactions, physical_entities,
                                                  relationships_with_index_style, components, entity_index_to_components_indices_mapping_list)

        # draw the histogram
        drawer = Drawer(num_of_edges, num_of_nodes, dimensionality, "All_data_in_Reactome")
        drawer.generate_histogram()

    def test(self):
        reactions = self.__reaction_processor.get_reactions_from_pathway('R-HSA-1640170')
        print(reactions)
        print(len(reactions))

        sum_of_physical_entities = 0
        for reaction in reactions:
            physical_entities = self.__reaction_processor.get_physical_entities_from_single_reaction(reaction)
            sum_of_physical_entities = sum_of_physical_entities + len(physical_entities)

        print("sum_of_physical_entities = " + str(sum_of_physical_entities))

        physical_entities_for_single_reaction = self.__reaction_processor.get_physical_entities_from_single_reaction(
            'R-HSA-8964492')

        physical_entities_unique = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(
            reactions)

        print(physical_entities_unique)

        print("sum_of_physical_entities_unique = " + str(len(physical_entities_unique)))

        component_ids_unique = self.__physical_entity_processor.get_unique_componets_from_set_of_physical_entities(
            physical_entities_unique)

        print("num of component_ids_unique = " + str(len(component_ids_unique)))

    def test_reactions(self):
        all_reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_in_Reactome()
        top_pathways = self.get_all_top_pathways()
        number_with_duplicate_elements = 0
        reactions_set = set()
        for top_pathway_id in top_pathways:
            reactions = self.__reaction_processor.get_reactions_from_pathway(top_pathway_id)
            number_with_duplicate_elements = number_with_duplicate_elements + len(reactions)
            for reaction in reactions:
                reactions_set.add(reaction)

        reactions_difference = set(all_reactions).difference(reactions_set)

        print("The num of reactions with duplicate elements are: " + str(number_with_duplicate_elements))
        print("The total num of reactions are: " + str(len(reactions_set)))

        for reaction_id in reactions_difference:
            print(reaction_id)


# one jump to n jump
# "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = 'R-HSA-9612973' AND n.speciesName='Homo sapiens' RETURN m"

# "MATCH (n:Reaction)-[r:input*1..]->(m:PhysicalEntity) WHERE n.stId = 'R-HSA-9626034' AND n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId"


class Drawer:
    def __init__(self, num_of_hyper_edges, num_of_nodes, dimensionality, pathway_name):
        self.num_of_hyper_edges = num_of_hyper_edges
        self.num_of_nodes = num_of_nodes
        self.dimensionality = dimensionality
        self.pathway_name = pathway_name
        self.path = "./data/" + pathway_name + "/"

    def generate_histogram(self):
        x1 = [self.pathway_name]
        y1 = [self.num_of_hyper_edges]
        y2 = [self.num_of_nodes]
        y3 = [self.dimensionality]
        name_of_file = self.pathway_name + ".html"
        path = self.path + '/' + name_of_file
        url = path + '/' + name_of_file

        if os.path.exists(url):
            print("file exists, we'll delete the original file \"" + name_of_file + "\", then create a new one")
            os.remove(url)

        bar = (
            Bar()
                .add_xaxis(x1)
                .add_yaxis("Hyper Edges(Reactions)", y1)
                .add_yaxis("Nodes(Physical Entity)", y2)
                .add_yaxis("Dimensionality(All components of nodes", y3)
                .set_global_opts(title_opts=opts.TitleOpts(title=self.pathway_name)))

        bar.render(path)


class FileProcessor:
    def __init__(self):
        self.filename_reactions = "edges.txt"
        self.filename_physical_entities = "nodes.txt"
        self.filename_relationships = "relationship.txt"
        self.filename_components_mapping = "components-mapping.txt"
        self.filename_components_all = "components-all.txt"

    # create the txt file to store the data
    def createFile(self, path, file_name):
        url = path + '/' + file_name
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(url):
            print("file exists, we'll delete the original file \"" + file_name + "\", then create a new one")
            os.remove(url)
        file = open(url, 'w', encoding='utf-8')

    # create the txt file to store the data with default path
    def createFileWithDefaultPath(self, file_name):
        path = './data'
        self.createFile(path, file_name)

    # write message to txt file
    def writeMessageToFile(self, path, file_name, message: list):
        url = path + '/' + file_name
        if not os.path.exists(url):
            print("error! the file \"" + file_name + "\" doesn't exist!")

        message = np.array(message)
        np.savetxt(url, message, delimiter=',', fmt='%s', encoding='utf-8')

        # file = open(url, "w")
        # for line in message:
        #     file.write(line+"\n")
        # file.close()

    # def writeMessageToFileTwoDimension(self):

    def execute_for_single_pathway(self, pathway_name, reaction_ids, physical_entity_ids,
                                   relationships_between_nodes_edges, component_ids, entity_component_mapping_list):

        path = "./data/" + pathway_name + "/"

        # write message to the file
        file_professor = FileProcessor()

        file_professor.createFile(path, self.filename_reactions)
        file_professor.createFile(path, self.filename_physical_entities)
        file_professor.createFile(path, self.filename_relationships)
        file_professor.createFile(path, self.filename_components_all)
        file_professor.createFile(path, self.filename_components_mapping)

        file_professor.writeMessageToFile(path, self.filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, self.filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, self.filename_relationships, relationships_between_nodes_edges)
        file_professor.writeMessageToFile(path, self.filename_components_all, component_ids)
        file_professor.writeMessageToFile(path, self.filename_components_mapping, entity_component_mapping_list)

    def read_file_via_lines(self, path, file_name):
        url = path + '/' + file_name
        file_handler = open(url, "r")

        res_list = []

        while (True):
            # Get next line from file
            line = file_handler.readline()

            # If the line is empty then the end of file reached
            if not line:
                break
            res_list.append(line)

        return res_list


if __name__ == '__main__':
    # graph = Graph("bolt://localhost:7687", auth=('neo4j', '123456'))
    #
    # processor = PhysicalEntityProcessor(graph)
    #
    # components = processor.get_components_of_physical_entity('R-HSA-170079')

    time_start = time.time()  # record the start time

    reactome_processor = ReactomeProcessor('neo4j', '123456')


    # R-HSA-9612973=
    # reactome_processor.execution_on_single_pathways("R-HSA-9612973")
    # reactome_processor.execution_on_single_pathways("R-HSA-1430728")

    # R-HSA-1640170
    # reactome_processor.execution_on_single_pathways("R-HSA-1640170")

    reactome_processor.execution_on_reactome()

    # reactome_processor.execution_on_all_pathways()

    # reactome_processor.test_reactions()

    time_end = time.time()  # record the ending time

    time_sum = time_end - time_start  # The difference is the execution time of the program in seconds

    print("success! it takes " + str(time_sum) + " seconds to extract the data from Reactome")
