# from neo4j import GraphDatabase
import numpy as np
from py2neo import Graph, Node, Relationship
import os
import time
from pyecharts import options as opts
from pyecharts.charts import Bar


# The processor which deals with the pathway
class PathWayProcessor:
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
        self.id_index = 0
        self.name_index = 1

    def get_reactions_from_pathway(self, pathway_stId):
        # if toplevel_pathway_stId == -1, it means we try to find reactions of homo sapiens for all the pathways
        if (-1 == pathway_stId):
            return self.get_all_reactions_of_homo_sapiens_in_Reactome()

        # we used to return "reaction_stId, reaction_displayName", but now only "reaction_stId"
        __instruction__ = "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = '" + str(
            pathway_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId"
        reactions = self.__graph.run(__instruction__).to_ndarray()
        reactions = reactions.flatten(order='C').tolist()
        # There'll be some duplicated reactions via our method in one pathway, so just reduce the duplicated ones
        reactions = list(set(reactions))

        # list of reaction_id
        return reactions

    def get_all_reactions_of_homo_sapiens_in_Reactome(self):
        reactions = self.__graph.run(
            "MATCH (n:Reaction) WHERE n.speciesName='Homo sapiens' RETURN n.stId").to_ndarray()
        reactions = reactions.flatten(order='C').tolist()
        return reactions

    def __get_input_relationship_edge_of_reaction(self, reaction_stId):
        __instruction_input__ = "MATCH (n:Reaction)-[r:input]->(m:PhysicalEntity) WHERE n.stId = '" + str(
            reaction_stId) + "' AND n.speciesName='Homo sapiens' RETURN m.stId, n.stId"
        __input_edges__ = self.__graph.run(__instruction_input__).to_ndarray()

        # get the line number of ndarray，which represents the number of input
        num_of_input_edges = __input_edges__.shape[0]

        # we build a complementary vertex based on the number of input
        supplement_vector = np.zeros(num_of_input_edges)

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
        if (__physical_entity_schemaClass__ndarray == None or __physical_entity_schemaClass__ndarray.size == 0):
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
        output: list of physical_entity_id(components)
        '''
    def get_unique_componets_from_set_of_physical_entities(self, physical_entities):

        component_ids_set = set()

        for physical_entity_id in physical_entities:
            component_ids = self.get_components_of_single_physical_entity(
                physical_entity_id)

            if len(component_ids) == 0:
                print("error! for finding no components -> physical_entity_stId:" + str(physical_entity_id))

            for component_id in component_ids:
                component_ids_set.add(component_id)

        component_ids_unique = list(component_ids_set)

        return component_ids_unique




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
        if(pathways.size == 1):
            return pathways[0]
        else:
            print("sorry, we can't find pathway with stId = '" + str(pathway_stId) + "'")
            return ''

    def get_reaction_name_by_id(self, reaction_stId):
        __instruction__ = "MATCH (n:Reaction) WHERE n.stId = '" + str(reaction_stId) + "' RETURN n.displayName"
        reactions = self.__graph.run(__instruction__).to_ndarray()
        reactions = reactions.flatten(order='C')
        if(reactions.size == 1):
            return reactions[0]
        else:
            print("sorry, we can't find reaction with stId = '" + str(reaction_stId) + "'")
            return ''

    def get_physical_entity_name_by_id(self, physical_entity_stId):
        __instruction__ = "MATCH (n:PhysicalEntity) WHERE n.stId = '" + str(physical_entity_stId) + "' RETURN n.displayName"
        physical_entities = self.__graph.run(__instruction__).to_ndarray()
        physical_entities = physical_entities.flatten(order='C')
        if(physical_entities.size == 1):
            return physical_entities[0]
        else:
            print("sorry, we can't find physical entity with stId = '" + str(physical_entity_stId) + "'")
            return ''

    def get_all_unique_edges_for_single_pathway(self, pathway_id):
        reactions = self.__reaction_processor.get_reactions_from_pathway(pathway_id)
        unique_edges_for_single_pathway = self.__reaction_processor.get_all_unique_edges_of_set_of_reactions(reactions)
        return unique_edges_for_single_pathway


    def extract_edges_nodes_dimensionality_for_one_pathway(self, pathway_stId):
        reactions = self.__reaction_processor.get_reactions_from_pathway(pathway_stId)
        print("************" + self.get_pathway_name_by_id(pathway_stId) + "************")

        # get unique physical entities for one pathway
        physical_entity_ids_from_reactions_for_one_pathway = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(
            reactions)

        # remove the duplicate components
        component_ids_unique_for_one_pathway = self.__physical_entity_processor.get_unique_componets_from_set_of_physical_entities(
            physical_entity_ids_from_reactions_for_one_pathway)

        num_of_edges = str(len(reactions))
        num_of_nodes = str(len(physical_entity_ids_from_reactions_for_one_pathway))
        dimensionality = str(len(component_ids_unique_for_one_pathway))

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        return reactions, physical_entity_ids_from_reactions_for_one_pathway, component_ids_unique_for_one_pathway


    def execution_on_single_pathways(self, pathway_stId):
        pathway_name = self.get_pathway_name_by_id(pathway_stId)

        reactions, physical_entity_ids, component_ids = self.extract_edges_nodes_dimensionality_for_one_pathway(pathway_stId)

        unique_edges_for_single_pathway = self.get_all_unique_edges_for_single_pathway(pathway_stId)

        # store data into a txt file
        file_processor = FileProcessor()
        file_processor.execute_for_single_path(pathway_name, reactions, physical_entity_ids, unique_edges_for_single_pathway)

        # draw the histogram
        drawer = Drawer(len(reactions), len(physical_entity_ids), len(component_ids), pathway_name)
        drawer.generate_histogram()

    def execution_on_all_pathways(self):
        top_pathways = self.get_all_top_pathways()
        for top_pathway_id in top_pathways:
            self.execution_on_single_pathways(top_pathway_id)

    def execution_on_reactome(self):
        reactions = self.__reaction_processor.get_all_reactions_of_homo_sapiens_in_Reactome()

        physical_entities = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(reactions)

        componets = self.__physical_entity_processor.get_unique_componets_from_set_of_physical_entities(physical_entities)

        num_of_edges = str(len(reactions))
        num_of_nodes = str(len(physical_entities))
        dimensionality = str(len(componets))

        print("************Reactome************")

        print("reactions(hyper edges): " + num_of_edges)
        print("physical entities(nodes): " + num_of_nodes)
        print("physical entities dimensionality(attributes): " + dimensionality)
        print("\n")

        if not os.path.exists("./data/AllReactome"):
            os.makedirs("./data/AllReactome")


        # draw the histogram
        drawer = Drawer(num_of_edges, num_of_nodes, dimensionality, "AllReactome")
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

        physical_entities_for_single_reaction = self.__reaction_processor.get_physical_entities_from_single_reaction('R-HSA-8964492')

        physical_entities_unique = self.__reaction_processor.get_unique_physical_entities_from_set_of_reactions(reactions)

        print(physical_entities_unique)

        print("sum_of_physical_entities_unique = " + str(len(physical_entities_unique)))

        component_ids_unique = self.__physical_entity_processor.get_unique_componets_from_set_of_physical_entities(physical_entities_unique)

        print("num of component_ids_unique = " + str(len(component_ids_unique)))





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
            .add_yaxis("Dimensionality", y3)
            .set_global_opts(title_opts=opts.TitleOpts(title=self.pathway_name)))

        bar.render(path)


class FileProcessor:
    def __init__(self):
        pass

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


    def execute_for_single_path(self, pathway_name, reaction_ids, physical_entity_ids, unique_edges_for_single_pathway):
        filename_reactions = "edges.txt"
        filename_physical_entities = "nodes.txt"
        filename_relationships = "relationship.txt"

        path = "./data/" + pathway_name + "/"

        # write message to the file
        file_professor = FileProcessor()

        file_professor.createFile(path, filename_reactions)
        file_professor.createFile(path, filename_physical_entities)
        file_professor.createFile(path, filename_relationships)

        file_professor.writeMessageToFile(path, filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, filename_relationships, unique_edges_for_single_pathway)

