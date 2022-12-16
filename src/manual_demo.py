import numpy as np
import torch
from dhg import Graph, Hypergraph
from dhg.models import GCN
from sklearn.metrics import ndcg_score

from data_loader import DataLoaderAttribute


@torch.no_grad()
def test(net_model, nodes_features, graph, labels, test_idx):
    net_model.eval()
    torch.set_printoptions(threshold=np.inf)

    nodes_features_value, nodes_features_indices = torch.topk(nodes_features, 15)
    print("input features")
    print(nodes_features_value[:3])
    print(nodes_features_indices[:3])
    print("\n")

    outs = net_model(nodes_features, graph)

    outs, labels = outs[test_idx], labels[test_idx]

    outs_value, outs_indices = torch.topk(outs, 5)
    print("output features")
    print(outs_value[:3])
    print(outs_indices[:3])
    print("\n")

    labels_value, labels_indices = torch.topk(labels, 5)
    print("labels features")
    print(labels_value[:3])
    print(labels_indices[:3])

    val_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())

    print(
        "\033[1;32m"
        + "The validation score is: "
        + "{:.5f}".format(val_res)
        + "\033[0m"
    )


if __name__ == '__main__':
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_loader = DataLoaderAttribute("Disease", "attribute prediction dataset")
    net_model = GCN(
        data_loader["num_features"], 32, data_loader["num_features"], use_bn=True
    )
    net_model.load_state_dict(torch.load("gcn.pkl"))
    test_mask = data_loader["test_node_mask"]

    # get the total number of nodes of this graph
    num_of_nodes: int = data_loader["num_nodes"]

    # generate the relationship between hyper edge and nodes
    # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
    hyper_edge_list = data_loader["edge_list"]

    # the hyper graph
    hyper_graph = Hypergraph(num_of_nodes, hyper_edge_list)

    # generate graph based on hyper graph
    graph = Graph.from_hypergraph_clique(hyper_graph, weighted=True)

    test_nodes_features = torch.FloatTensor(data_loader["test_nodes_features"])

    labels = torch.FloatTensor(data_loader["raw_nodes_features"])

    test_nodes_features, labels = (
        test_nodes_features.to(device),
        labels.to(device)
    )
    graph = graph.to(device)
    net_model = net_model.to(device)

    test(net_model, test_nodes_features, graph, labels, test_mask)

    print("finish")

