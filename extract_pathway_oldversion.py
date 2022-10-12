# from neo4j import GraphDatabase
import numpy as np
from py2neo import Graph, Node, Relationship
import os
import time

# 老版本的代码是不行的，因为有一个严重的bug，np.unique() 不能对二维的数组去重，必须要遍历转化成tuple，然后放进set中去重，或者是转化成imaginary number去重
# 所以oldversion的pgysical_entity 是没有去重完全的
# 新版本解决了这一问题


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


class PhysicalEntityProcessor:

    def __init__(self, graph: Graph):
        self.__graph = graph
        self.__type_detector = TypeDetectorOfPhysicalEntity()
        self.__funcs_for_get_components = {'Complex': self.__get_components_of_complex,
                                         'Polymer': self.__get_components_of_polymer,
                                         'GenomeEncodedEntity': self.__get_components_of_GenomeEncodedEntity,
                                         'EntitySet': self.__get_components_of_EntitySet,
                                         'SimpleEntity': self.__get_components_of_SimpleEntity,
                                         'OtherEntity': self.__get_components_of_OtherEntity,
                                         'Drug': self.__get_components_of_Drug}

    def __get_components_of_complex(self, physical_entity_id: str):
        # MATCH (n:Complex)-[r:hasComponent]->(m) WHERE n.stId = 'R-HSA-917704' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:Complex)-[r:hasComponent]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId, m.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()

        # if Complex has no components, we define itself as a component.
        if (__components__.size == 0):
            # MATCH (n:Complex) WHERE n.stId = 'R-HSA-917704' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:Complex) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId, n.displayName"
            __components__ = self.__graph.run(__instruction__).to_ndarray()

        return __components__

    def __get_components_of_polymer(self, physical_entity_id: str):
        # MATCH (n:Polymer)-[r:repeatedUnit]->(m) WHERE n.stId = 'R-HSA-9626247' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:Polymer)-[r:repeatedUnit]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId, m.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()

        # if Polymer has no components, we define itself as a component.
        if (__components__.size == 0):
            # MATCH (n:Polymer) WHERE n.stId = 'R-HSA-2214302' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:Polymer) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId, n.displayName"
            __components__ = self.__graph.run(__instruction__).to_ndarray()

        return __components__

    # GenomeEncodedEntity has no components, so define itself as an attribute
    def __get_components_of_GenomeEncodedEntity(self, physical_entity_id: str):
        # MATCH (n:GenomeEncodedEntity) WHERE n.stId = 'R-HSA-2029007' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:GenomeEncodedEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId, n.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()
        return __components__

    def __get_components_of_EntitySet(self, physical_entity_id: str):
        # MATCH (n:EntitySet)-[r:hasMember]->(m) WHERE n.stId = 'R-HSA-170079' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName
        __instruction__ = "MATCH (n:EntitySet)-[r:hasMember]->(m) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN m.stId, m.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()

        # if EntitySet has no members(component), we define itself as a member(component).
        if (__components__.size == 0):
            # MATCH (n:EntitySet) WHERE n.stId = 'R-HSA-170079' AND n.speciesName='Homo sapiens' RETURN n.stId, n.displayName
            __instruction__ = "MATCH (n:EntitySet) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.stId, n.displayName"
            __components__ = self.__graph.run(__instruction__).to_ndarray()

        return __components__

    # SimpleEntity has no components, so define itself as an attribute
    # A kind reminder that the SimpleEntity has no n.speciesName attribute
    def __get_components_of_SimpleEntity(self, physical_entity_id: str):
        # MATCH (n:SimpleEntity) WHERE n.stId = 'R-ALL-29438' RETURN n.stId, n.displayName, n.speciesName
        __instruction__ = "MATCH (n:SimpleEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId, n.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()
        return __components__

    # OtherEntity has no components, so define itself as an attribute
    # A kind reminder that the OtherEntity has no n.speciesName attribute
    def __get_components_of_OtherEntity(self, physical_entity_id: str):
        # MATCH (n:OtherEntity) WHERE n.stId = 'R-ALL-422139' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:OtherEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId, n.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()
        return __components__

    # Drug has no components, so define itself as an attribute
    # A kind reminder that the Drug has no n.speciesName attribute
    def __get_components_of_Drug(self, physical_entity_id: str):
        # MATCH (n:Drug) WHERE n.stId = 'R-ALL-9674322' RETURN n.stId, n.displayName
        __instruction__ = "MATCH (n:Drug) WHERE n.stId = '" + str(
            physical_entity_id) + "' RETURN n.stId, n.displayName"
        __components__ = self.__graph.run(__instruction__).to_ndarray()
        return __components__

    def __get_schemaClass_of_physical_entity(self, physical_entity_id: str):
        __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
            physical_entity_id) + "' AND n.speciesName='Homo sapiens' RETURN n.schemaClass"
        __physical_entity_schemaClass__ndarray = self.__graph.run(__instruction__).to_ndarray()
        # if is None, it's possibly because that the node is An Simple Entity/OtherEntity/Drug which doesn't have n.speciesName
        if(__physical_entity_schemaClass__ndarray == None or __physical_entity_schemaClass__ndarray.size == 0):
            __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(
                physical_entity_id) + "' RETURN n.schemaClass"
            __physical_entity_schemaClass__ndarray = self.__graph.run(__instruction__).to_ndarray()

        __physical_entity_schemaClass = __physical_entity_schemaClass__ndarray[0, 0]

        return __physical_entity_schemaClass

    def get_components_of_physical_entity(self, physical_entity_id: str):
        __physical_entity_schemaClass__ = self.__get_schemaClass_of_physical_entity(physical_entity_id)
        type_of_physical_entity = self.__type_detector.get_type_of_physical_entity(__physical_entity_schemaClass__)
        func_for_get_components_selected = self.__funcs_for_get_components.get(type_of_physical_entity)
        components = []
        if(func_for_get_components_selected is not None):
            components = func_for_get_components_selected(physical_entity_id)
        # id, name
        return components


class ReactomeProcessor:
    def __init__(self):
        self.__link = "bolt://localhost:7687"
        self.__user_name = 'neo4j'
        self.__password = '123456'
        self.__graph = self.__login(self.__link, self.__user_name, self.__password)
        # PhysicalEntityProcessor
        self.__physical_entity_processor = PhysicalEntityProcessor(self.__graph)

        self.id_index = 0
        self.name_index = 1



    @staticmethod
    def __login(link, user_name, password):
        graph = Graph(link, auth=(user_name, password))
        return graph

    def get_all_top_level_pathways(self):
        # TopLevelPathway
        # "MATCH (n:TopLevelPathway) WHERE n.speciesName='Homo sapiens' RETURN n.schemaClass"
        # "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway', 'Pathway']) AND n.speciesName='Homo sapiens' RETURN n LIMIT 20"
        __instruction__ = "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway']) AND n.speciesName='Homo sapiens' RETURN n.displayName, n.stId"
        __toplevel_pathways = self.__graph.run(__instruction__).to_ndarray()
        return __toplevel_pathways

    def get_reactions_from_toplevel_pathway(self, toplevel_pathway_stId):
        # if toplevel_pathway_stId == -1, it means we try to find reactions for all the pathways
        if(-1 == toplevel_pathway_stId):
            reactions = self.__graph.run("MATCH (n:Reaction) WHERE n.speciesName='Homo sapiens' RETURN n.displayName, n.stId").to_ndarray()
            return reactions
        __instruction__ = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = '" + str(
            toplevel_pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName"
        __reactions__ = self.__graph.run(__instruction__).to_ndarray()
        __reactions__ = np.unique()
        return __reactions__

    def get_all_reactions_of_homo_sapiens(self):
        __reactions__ = self.get_reactions_from_toplevel_pathway(-1)
        return __reactions__

    def __get_input_edges_for_reaction(self, reaction_stId):
        __instruction_input__ = "MATCH (n:Reaction)-[r:input*1..]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName, n.stId, n.displayName"
        __input_edges__ = self.__graph.run(__instruction_input__).to_ndarray()

        # get the line number of ndarray，which represents the number of input
        num_of_input_edges = __input_edges__.shape[0]

        # we build a complementary vertex based on the number of input
        supplement_vector = np.zeros(num_of_input_edges)

        # we add a new column, then fill it with 0, which represents input
        __input_edges__ = np.insert(__input_edges__, 4, supplement_vector, axis=1)

        # PhysicalEntity_id, PhysicalEntity_name, Reaction_id, Reaction_name, 0
        return __input_edges__

    def __get_output_edges_for_reaction(self, reaction_stId):

        __instruction_output__ = "MATCH (n:Reaction)-[r:output*1..]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, m.displayName, n.stId, n.displayName"
        __output_edges__ = self.__graph.run(__instruction_output__).to_ndarray()

        # get the line number of ndarray，which represents the number of input
        __num_of_output_edges = __output_edges__.shape[0]

        # we build a complementary vertex based on the number of input
        supplement_vector = np.ones(__num_of_output_edges)

        # we add a new column, then fill it with 0, which represents input
        __output_edges__ = np.insert(__output_edges__, 4, supplement_vector, axis=1)

        # PhysicalEntity_id, PhysicalEntity_name, Reaction_id, Reaction_name, 1
        return __output_edges__

    def __get_all_edges_for_reaction(self, reaction_stId):
        __input_edges__ = self.__get_input_edges_for_reaction(reaction_stId)
        __output_edges__ = self.__get_output_edges_for_reaction(reaction_stId)
        __edges_for_reaction = np.vstack((__input_edges__, __output_edges__))

        # PhysicalEntity_id, PhysicalEntity_name, Reaction_id, Reaction_name, 0/1
        return __edges_for_reaction

    def __get_physical_entities_from_reaction(self, reaction_stId):
        __edges_for_reaction = self.__get_all_edges_for_reaction(reaction_stId)
        # __edges_unique = np.unique(__edges, axis=0)

        # PhysicalEntity_id, PhysicalEntity_name
        __physical_entities_from_reaction = __edges_for_reaction[:, [0, 1]]

        # __physical_entities_unique = np.unique(__edges_for_reaction, axis=0)
        # print("We'll print the physical entities for reaction:\n")
        # print(__physical_entities_from_reaction)

        # PhysicalEntity_id, PhysicalEntity_name
        return __physical_entities_from_reaction

    def __get_physical_entities_from_set_of_reactions(self, reactions):
        sum_of_entities = 0

        # PhysicalEntity_id, PhysicalEntity_name
        physical_entities_from_set_of_reactions = np.empty(shape=(0, 2))

        for reaction in reactions:
            reaction_stId = reaction[self.id_index]
            # PhysicalEntity_id, PhysicalEntity_name
            physical_entities = self.__get_physical_entities_from_reaction(reaction_stId)
            sum_of_entities += physical_entities.shape[0]
            physical_entities_from_set_of_reactions = np.vstack((physical_entities_from_set_of_reactions, physical_entities))

        # remove the duplicate physical entities
        physical_entities_for_one_pathway = np.unique(physical_entities_from_set_of_reactions)

    def execute(self):
        __toplevel_pathways = self.get_all_top_level_pathways()
        for toplevel_pathway in __toplevel_pathways:
            toplevel_pathway_stId = toplevel_pathway[1]
            reactions = self.get_reactions_from_toplevel_pathway(toplevel_pathway_stId)
            print("************" + toplevel_pathway[0] + "************")
            print("reactions(hyper graph):" + str(reactions.shape[0]))
            sum_of_entities = 0

            # PhysicalEntity_id, PhysicalEntity_name
            physical_entities_for_one_pathway = np.empty(shape=(0, 2))

            # id, name
            components_for_one_pathway = np.empty(shape=(0, 2))

            for reaction in reactions:
                reaction_stId = reaction[self.id_index]
                # PhysicalEntity_id, PhysicalEntity_name
                physical_entities = self.__get_physical_entities_from_reaction(reaction_stId)
                sum_of_entities += physical_entities.shape[0]
                physical_entities_for_one_pathway = np.vstack((physical_entities_for_one_pathway, physical_entities))

            # remove the duplicate physical entities
            physical_entities_for_one_pathway = np.unique(physical_entities_for_one_pathway)

            for physical_entity_for_one_pathway in physical_entities_for_one_pathway:
                components = self.__physical_entity_processor.get_components_of_physical_entity(physical_entity_for_one_pathway[self.id_index])

                if components.size == 0:
                    print("error: toplevel_pathway_stId: " + str(toplevel_pathway_stId) + "physical_entity_stId: " + str(physical_entity_for_one_pathway[self.id_index]))

                components_for_one_pathway = np.vstack((components_for_one_pathway, components))

            # remove the duplicate components
            components_unique_for_one_pathway = np.unique(components_for_one_pathway)

            print("physical entities(nodes):" + str(sum_of_entities))
            print("physical entities(nodes):" + str(physical_entities_for_one_pathway.shape[0]))
            print("physical entities dimensionality: " + str(components_unique_for_one_pathway.shape[0]))
            print("\n\n")

    def test(self):

        reactions = self.get_reactions_from_toplevel_pathway('R-HSA-1640170')
        print(reactions)
        print(reactions.shape[0])

        # sum_of_physical_entities = 0
        # for reaction in reactions:
        #     physical_entities = self.__get_physical_entities_from_reaction(reaction)
        #     sum_of_physical_entities = sum_of_physical_entities + len(physical_entities)
        #
        # print("sum_of_physical_entities = " + str(sum_of_physical_entities))
        #
        # physical_entities_for_single_reaction = self.__reaction_processor.get_physical_entities_from_single_reaction(
        #     'R-HSA-8964492')
        #
        # physical_entities_unique = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(
        #     reactions)
        #
        # print(physical_entities_unique)
        #
        # print("sum_of_physical_entities_unique = " + str(len(physical_entities_unique)))
        #
        # component_ids_unique = self.__physical_entity_processor.get_unique_componets_from_set_of_physical_entities(
        #     physical_entities_unique)
        #
        # print("num of component_ids_unique = " + str(len(component_ids_unique)))
        # self.__physical_entity_processor.get_components_of_physical_entity('R-ALL-9754127')


if __name__ == '__main__':
    processor = ReactomeProcessor()
    processor.test()





# print(toplevel_pathways[0, 1])

# print(toplevel_pathways)


# "MATCH (n) WHERE any(label in labels(n) WHERE label in ['TopLevelPathway', 'Pathway']) AND n.speciesName='Homo sapiens' RETURN n LIMIT 20"

# 一跳 到 N跳
"MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = 'R-HSA-9612973' AND n.speciesName='Homo sapiens' RETURN m"

"MATCH (n:Reaction)-[r:input*1..]->(m:PhysicalEntity) WHERE n.stId = 'R-HSA-9612973' AND n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId"

# "MATCH (n:Reac‘tion)-[r:input*1..]->(m:PhysicalE  ntity) WHERE n.stId = 'R-HSA-9626034' AND n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId"
