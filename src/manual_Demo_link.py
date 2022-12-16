import copy
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.models import GCN
from sklearn.metrics import ndcg_score, accuracy_score

import utils
from data_loader import DataLoaderLink

@torch.no_grad()
def test(
    net_model,
    nodes_features,
    test_hyper_edge_list: list[list[int]],
    graph,
    labels,
    test_idx,
):
    net_model.eval()
    torch.set_printoptions(threshold=np.inf)

    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            test_hyper_edge_list, nodes_features
        )
    )

    print("input features")
    test_slice = test_idx[:3]
    test_hyper_edge_list_slice = [test_hyper_edge_list[i] for i in test_slice]
    print(test_hyper_edge_list_slice)
    print("\n")

    edges_embeddings = edges_embeddings.to(device)

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    outs, labels = outs[test_idx], labels[test_idx]

    outs_value, outs_indices = torch.topk(outs, 15)
    print("output features")
    print(outs_value[:3])
    print(outs_indices[:3])
    print("\n")

    labels_value, labels_indices = torch.topk(labels, 10)
    print("labels features")
    print(labels_value[:3])
    print(labels_indices[:3])



    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())

    print(
        "\n\033[1;35m"
        + "The final test score is: "
        + "{:.5f}".format(ndcg_res)
        + "\033[0m"
    )



def main(dataset: str, task: str):
    """
    This method is for the whole train process
    :param dataset: The name of dataset, ex. Disease, Immune System, Metabolism, Signal Transduction
    :param task: The name of task, ex. input link prediction dataset, output link prediction dataset
    :return:
    """
    # set device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialize the data_loader
    # data_loader = DataLoaderLink("Disease", "input link prediction dataset")
    data_loader = DataLoaderLink(dataset, task)

    # get the total number of nodes of this graph
    num_of_nodes: int = data_loader["num_nodes"]

    num_of_edges: int = data_loader["num_edges"]

    # get the raw, train,val,test nodes features
    raw_nodes_features = torch.FloatTensor(data_loader["raw_nodes_features"])
    train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])
    validation_nodes_features = torch.FloatTensor(
        data_loader["validation_nodes_features"]
    )
    test_nodes_features = torch.FloatTensor(data_loader["test_nodes_features"])

    # generate the relationship between hyper edge and nodes
    # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
    raw_hyper_edge_list = data_loader["raw_edge_list"]
    train_hyper_edge_list = data_loader["train_edge_list"]
    validation_hyper_edge_list = data_loader["validation_edge_list"]
    test_hyper_edge_list = data_loader["test_edge_list"]

    # get train, validation, test mask to track the nodes
    train_edge_mask = data_loader["train_edge_mask"]
    val_edge_mask = data_loader["val_edge_mask"]
    test_edge_mask = data_loader["test_edge_mask"]

    # get the labels - the original nodes features
    labels = torch.FloatTensor(
        utils.encode_edges_features(raw_hyper_edge_list, num_of_edges, num_of_nodes)
    )

    # the train hyper graph
    hyper_graph = Hypergraph(num_of_nodes, copy.deepcopy(train_hyper_edge_list))

    # generate train graph based on hyper graph
    graph = Graph.from_hypergraph_clique(hyper_graph, weighted=True)

    # the GCN model
    net_model = GCN(
        data_loader["num_features"], 32, data_loader["num_features"], use_bn=True
    )

    net_model.load_state_dict(torch.load("gcn_link.pkl"))

    # set the device
    raw_nodes_features = raw_nodes_features.to(device)
    # train_nodes_features, validation_nodes_features, test_nodes_features, labels = train_nodes_features.to(
    #     device), validation_nodes_features.to(device), test_nodes_features.to(device), labels.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    net_model = net_model.to(device)

    print("GCN Baseline")

    test(
        net_model,
        raw_nodes_features,
        test_hyper_edge_list,
        graph,
        labels,
        test_edge_mask,
    )


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    main("Disease", "input link prediction dataset")

