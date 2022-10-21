# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import torch
from extract_pathway import PathWayProcessor, ReactionProcessor, PhysicalEntityProcessor, ReactomeProcessor
from py2neo import Graph, Node, Relationship
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def print_top_path_id_and_name(reactome_processor: ReactomeProcessor):
    toplevel_pathways = reactome_processor.get_all_top_pathways()

    for toplevel_pathway_id in toplevel_pathways:
        print(toplevel_pathway_id + "    " + reactome_processor.get_pathway_name_by_id(toplevel_pathway_id))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # graph = Graph("bolt://localhost:7687", auth=('neo4j', '123456'))
    #
    # processor = PhysicalEntityProcessor(graph)
    #
    # components = processor.get_components_of_physical_entity('R-HSA-170079')

    time_start = time.time()  # record the start time

    reactome_processor = ReactomeProcessor('neo4j', '123456')

    # reactome_processor.execution_on_single_pathways("R-HSA-1430728")

    # R-HSA-1640170
    # reactome_processor.execution_on_single_pathways("R-HSA-1640170")


    # reactome_processor.execution_on_reactome()

    reactome_processor.execution_on_all_pathways()


    time_end = time.time()  # record the ending time

    time_sum = time_end - time_start  # The difference is the execution time of the program in seconds

    print("success! it takes " + str(time_sum) + " seconds to extract the data from Reactome")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


