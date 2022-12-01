import copy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.models import GCN
from sklearn.metrics import ndcg_score, accuracy_score

import utils
from data_loader import DataLoaderLink

learning_rate = 0.01
weight_decay = 5e-4


def train(net_model: torch.nn.Module, nodes_features: torch.Tensor, train_hyper_edge_list: list[list[int]],
          graph: Graph, labels: torch.Tensor,
          train_idx: list[bool],
          optimizer: optim.Adam, epoch: int):
    net_model.train()

    st = time.time()
    optimizer.zero_grad()

    edges_embeddings = utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(train_hyper_edge_list,
                                                                                              nodes_features)

    edges_embeddings = edges_embeddings.to(device)

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    outs, labels = outs[train_idx], labels[train_idx]

    loss = F.cross_entropy(outs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def validation(net_model, nodes_features, validation_hyper_edge_list: list[list[int]], graph, labels, validation_idx):
    net_model.eval()

    edges_embeddings = utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(validation_hyper_edge_list,
                                                                                              nodes_features)

    edges_embeddings = edges_embeddings.to(device)

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    outs, labels = outs[validation_idx], labels[validation_idx]
    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)

    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)

    print("\033[1;32m" + "The validation score is: " + "{:.5f}".format(ndcg_res) + "\033[0m")
    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )
    return ndcg_res, acc_res


@torch.no_grad()
def test(net_model, nodes_features, test_hyper_edge_list: list[list[int]],  graph, labels, test_idx):
    net_model.eval()

    edges_embeddings = utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
        test_hyper_edge_list,
        nodes_features)

    edges_embeddings = edges_embeddings.to(device)

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    outs, labels = outs[test_idx], labels[test_idx]
    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)

    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)

    print("\n\033[1;35m" + "The final test score is: " + "{:.5f}".format(ndcg_res) + "\033[0m")

    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )

    return ndcg_res, acc_res


if __name__ == '__main__':
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialize the data_loader
    data_loader = DataLoaderLink("Disease", "input link prediction dataset")
    # data_loader = DataLoaderLink("Disease", "output link prediction dataset")

    # get the total number of nodes of this graph
    num_of_nodes: int = data_loader["num_nodes"]

    num_of_edges: int = data_loader["num_edges"]

    # get the raw, train,val,test nodes features
    raw_nodes_features = torch.FloatTensor(data_loader["raw_nodes_features"])
    train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])
    validation_nodes_features = torch.FloatTensor(data_loader["validation_nodes_features"])
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
    labels = torch.FloatTensor(utils.encode_edges_features(raw_hyper_edge_list, num_of_edges, num_of_nodes))

    # the train hyper graph
    hyper_graph_train = Hypergraph(num_of_nodes, copy.deepcopy(train_hyper_edge_list))

    # generate train graph based on hyper graph
    graph_train = Graph.from_hypergraph_clique(hyper_graph_train, weighted=True)

    # the train hyper graph
    hyper_graph_validation = Hypergraph(num_of_nodes, copy.deepcopy(train_hyper_edge_list))

    # generate train graph based on hyper graph
    graph_validation = Graph.from_hypergraph_clique(hyper_graph_validation, weighted=True)

    # the train hyper graph
    hyper_graph_test = Hypergraph(num_of_nodes, copy.deepcopy(train_hyper_edge_list))

    # generate train graph based on hyper graph
    graph_test = Graph.from_hypergraph_clique(hyper_graph_test, weighted=True)



    # the GCN model
    net_model = GCN(data_loader["num_features"], 32, data_loader["num_features"], use_bn=True)

    # set the optimizer
    optimizer = optim.Adam(net_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # set the device
    train_nodes_features, validation_nodes_features, test_nodes_features, labels = train_nodes_features.to(
        device), validation_nodes_features.to(device), test_nodes_features.to(device), labels.to(device)
    graph_train = graph_train.to(device)
    graph_validation = graph_validation.to(device)
    graph_test = graph_test.to(device)
    net_model = net_model.to(device)


    print("GCN Baseline")

    # start to train
    for epoch in range(200):
        # train
        # call the train method
        train(net_model, train_nodes_features, train_hyper_edge_list, graph_train, labels, train_edge_mask, optimizer, epoch)

        if epoch % 1 == 0:
            with torch.no_grad():
                # validation(net_model, validation_nodes_features, validation_hyper_edge_list, graph_validation, labels, val_edge_mask)
                validation(net_model, validation_nodes_features, validation_hyper_edge_list, graph_validation, labels,
                           val_edge_mask)

    test(net_model, test_nodes_features, test_hyper_edge_list, graph_test, labels, test_edge_mask)
