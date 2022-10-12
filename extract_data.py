# from neo4j import GraphDatabase
import numpy as np
from py2neo import Graph, Node, Relationship
import os
import time
import shutil  # 用于删除


# from dhg.data import Cooking200


time_start = time.time()  # 记录开始时间

# create the txt file to store the data
def createFile(path, file_name):
    url = path + '/' + file_name
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(url):
        print("file exists, we'll delete the original file \"" + file_name + "\", then create a new one")
        os.remove(url)
    file = open(url, 'w', encoding='utf-8')

# create the txt file to store the data with default path
def createFileWithDefaultPath(file_name):
    path = './data'
    createFile(path, file_name)

# write message to txt file
def writeMessageToFile(path, file_name, message):
    url = path + '/' + file_name
    if not os.path.exists(url):
        print("error! the file \"" + file_name + "\" doesn't exist!")
    # np.savetxt(url, message)
    np.savetxt(url, message, delimiter=',', fmt='%s', encoding='utf-8')
    # np.savetxt(url, message, fmt='%s')

# graph = Graph("http://local:7474", auth=('neo4j', '123456'))
# 这里用http就是不停的报错，用bolt连接就好，这里是一个大坑!
graph = Graph("bolt://localhost:7687", auth=('neo4j', '123456'))

# test = graph.run("MATCH (n:BlackBoxEvent) RETURN n LIMIT 25").to_data_frame()
# WHERE n.speciesName='Homo sapiens' means we only consider human beings

'''
"MATCH (n:PhysicalEntity) WHERE n.speciesName='Homo sapiens' RETURN n.displayName, n.stId LIMIT 25"
'''
physical_entities = \
    graph.run(
        "MATCH (n:PhysicalEntity) WHERE n.speciesName='Homo sapiens' RETURN n.displayName, n.stId").to_ndarray()
print("physical_entities:\n")
print(physical_entities)
print("\n**********************\n")

'''
"MATCH (n:Reaction) WHERE n.speciesName='Homo sapiens' RETURN n.displayName, n.stId LIMIT 25"
'''
reactions = \
    graph.run(
        "MATCH (n:Reaction) WHERE n.speciesName='Homo sapiens' RETURN n.displayName, n.stId").to_ndarray()

print("reactions:\n")
# print(reactions)
print("\n**********************\n")

# 一定要注意，是Reaction 选择自己的 input 和 output，如下所示，m:PhysicalEntity 是 n:Reaction的输入
# Reaction 和 我们的 PhysicalEntity 是多对多的关系

# physicalEntity name, physicalEntity id, reaction name, reaction id, 0 (the physicalEntity is input for reaction)
'''
"MATCH p=(n:Reaction)-[r:input]->(m:PhysicalEntity) WHERE n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId LIMIT 25"
'''
input_edges = \
    graph.run(
        "MATCH p=(n:Reaction)-[r:input]->(m:PhysicalEntity) WHERE n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId").to_ndarray()

'''
拿到 input_edges 的行数，就是我们输入边的条数，每一行的最后都要插入一个0来表明是input关系
最后一列添加一列 0 来表征 是input关系
'''
# 拿到 ndarray 的 行数，表示一共有多少个输入input
num_of_input_edges = input_edges.shape[0]
# 根据input的个数，构造一个 全为0的补充向量
supplement_vector = np.zeros(num_of_input_edges)
# 将补充向量 添加到 每一行的最后一列，相当于 是在 每一行的最后一列 都增加了一个0
input_edges = np.insert(input_edges, 4, supplement_vector, axis=1)

print("input_edges:\n")
# print(input_edges)
print("\n**********************\n")

output_edges = \
    graph.run(
        "MATCH p=(n:Reaction)-[r:output]->(m:PhysicalEntity) WHERE n.speciesName='Homo sapiens' RETURN m.displayName, m.stId, n.displayName, n.stId LIMIT 25").to_ndarray()

'''
拿到 output_edges 的行数，就是我们输出边的条数，每一行的最后都要插入一个1来表明是output关系
最后一列添加一列 0 来表征 是input关系
'''
# 拿到 ndarray 的 行数，表示一共有多少个输出output
num_of_output_edges = output_edges.shape[0]
# 根据output的个数，构造一个 全为1的补充向量
supplement_vector = np.ones(num_of_output_edges)
# 将补充向量 添加到 每一行的最后一列，相当于 是在 每一行的最后一列 都增加了一个1
output_edges = np.insert(output_edges, 4, supplement_vector, axis=1)

print("output_edges:\n")
# print(output_edges)
print("\n**********************\n")

edges = np.vstack((input_edges, output_edges))

print("edges:\n")
# print(edges)

filename_physical_entities = "physical_entities.txt"
filename_reactions = "reactions.txt"
filename_edges = "edges.txt"

createFileWithDefaultPath(filename_physical_entities)
createFileWithDefaultPath(filename_reactions)
createFileWithDefaultPath(filename_edges)

path = "./data"
writeMessageToFile(path, filename_physical_entities, physical_entities)
writeMessageToFile(path, filename_reactions, reactions)
writeMessageToFile(path, filename_edges, edges)






time_end = time.time()    # 记录结束时间

time_sum = time_end - time_start      # 计算的时间差为程序的执行时间，单位为秒/s

print("success! We write all the datas to three different files \"" + filename_physical_entities + "\"" + ", \"" + filename_reactions + "\", \"" + filename_edges + "\", it takes " + str(time_sum) + " seconds")

print("There are " + str(physical_entities.shape[0]) + " nodes(physical entity), and " + str(reactions.shape[0]) + " hyper edges in this hyper graph.")

# 一跳 到 N跳
# "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = 'R-HSA-9612973' AND n.speciesName='Homo sapiens' RETURN m"

# "MATCH (n:Pathway)-[r:hasEvent*1..]->(m:Reaction) WHERE n.stId = 'R-HSA-9612973' AND n.speciesName='Homo sapiens' RETURN m"


