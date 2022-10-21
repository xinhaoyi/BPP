import os
import numpy as np


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
            print("file exists, we'll delete the original file \"" + url + "\", then create a new one")
            os.remove(url)
        file = open(url, 'w', encoding='utf-8')

    def delete_file(self, path, file_name) -> None:
        url = path + '/' + file_name
        if os.path.exists(url):
            os.remove(url)

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
                                   unique_edges_for_single_pathway, component_ids, components_dic):

        path = "./data/" + pathway_name + "/"

        # write message to the file
        file_professor = FileProcessor()

        # dela with the components message like -
        # a list of ["node_id:component_id,component_id,component_id..", "node_id:component_id,component_id,component_id.."]
        component_msg_list = []
        for node_id, set_of_components in components_dic.items():
            component_msg = node_id + ":"
            for component_id in set_of_components:
                component_msg = component_msg + component_id + ","
            # remove the comma in the end
            component_msg = component_msg[:-1]
            component_msg_list.append(component_msg)

        file_professor.createFile(path, self.filename_reactions)
        file_professor.createFile(path, self.filename_physical_entities)
        file_professor.createFile(path, self.filename_relationships)
        file_professor.createFile(path, self.filename_components_all)
        file_professor.createFile(path, self.filename_components_mapping)

        file_professor.writeMessageToFile(path, self.filename_reactions, reaction_ids)
        file_professor.writeMessageToFile(path, self.filename_physical_entities, physical_entity_ids)
        file_professor.writeMessageToFile(path, self.filename_relationships, unique_edges_for_single_pathway)
        file_professor.writeMessageToFile(path, self.filename_components_all, component_ids)
        file_professor.writeMessageToFile(path, self.filename_components_mapping, component_msg_list)

    def read_file_via_lines(self, path, file_name):
        url = path + '/' + file_name
        file_handler = open(url, "r")

        res_list = []

        while (True):
            # Get next line from file
            line = file_handler.readline()
            line = line.replace('\r', '').replace('\n', '').replace('\t', '')

            # If the line is empty then the end of file reached
            if not line:
                break
            res_list.append(line)

        return res_list