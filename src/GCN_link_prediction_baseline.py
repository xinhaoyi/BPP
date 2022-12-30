import copy
import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.models import GCN
from sklearn.metrics import ndcg_score, accuracy_score

import utils
from data_loader import DataLoaderLink

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

learning_rate = 0.01
weight_decay = 5e-4


def train(
    net_model: torch.nn.Module,
    nodes_features: torch.Tensor,
    train_hyper_edge_list: list[list[int]],
    graph: Graph,
    labels: torch.Tensor,
    train_idx: list[bool],
    optimizer: optim.Adam,
    epoch: int,
):
    net_model.train()

    st = time.time()
    optimizer.zero_grad()

    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            train_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(device)
    edges_embeddings = edges_embeddings[train_idx]

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    loss = F.cross_entropy(outs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def validation(
    net_model,
    nodes_features,
    validation_hyper_edge_list: list[list[int]],
    graph,
    labels,
    validation_idx,
):
    net_model.eval()

    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            validation_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(device)
    nodes_embeddings = net_model(nodes_features, graph)
    # edges_embeddings = edges_embeddings[validation_idx]

    # torch.backends.cudnn.enabled = False
    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    # outs = [[0.1, 0.9, 0.3, 0.9],[0.1, 0.2, 0.3, 0.9]]
    # labels = [[0, 1, 0, 1], [0, 0, 0, 1]]
    # outs, labels = outs[validation_idx], labels[validation_idx]
    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)

    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)

    print(
        "\033[1;32m"
        + "The validation score is: "
        + "{:.5f}".format(ndcg_res)
        + "\033[0m"
    )
    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )
    return ndcg_res, acc_res


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

    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            test_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(device)
    nodes_embeddings = net_model(nodes_features, graph)
    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())
    # edges_embeddings = edges_embeddings[validation_idx]

    # outs, labels = outs[test_idx], labels[test_idx]
    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)

    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)

    print(
        "\n\033[1;35m"
        + "The final test score is: "
        + "{:.5f}".format(ndcg_res)
        + "\033[0m"
    )

    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )

    return ndcg_res, acc_res


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

    # get the raw, train,val,test nodes features
    train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])

    # generate the relationship between hyper edge and nodes
    # ex. [[1,2,3,4], [3,4], [9,7,4]...] where [1,2,3,4] represent a hyper edge
    train_hyper_edge_list = data_loader["train_edge_list"]
    validation_hyper_edge_list = data_loader["validation_edge_list"]
    test_hyper_edge_list = data_loader["test_edge_list"]

    # get train, validation, test mask to track the nodes
    train_edge_mask = data_loader["train_edge_mask"]
    val_edge_mask = data_loader["val_edge_mask"]
    test_edge_mask = data_loader["test_edge_mask"]

    train_labels = data_loader["train_labels"]
    test_labels = data_loader["test_labels"]
    validation_labels = data_loader["validation_labels"]

    # the train hyper graph
    hyper_graph = Hypergraph(num_of_nodes, copy.deepcopy(train_hyper_edge_list))

    # generate train graph based on hyper graph
    graph = Graph.from_hypergraph_clique(hyper_graph, weighted=True)

    # the GCN model
    net_model = GCN(
        data_loader["num_features"], 32, data_loader["num_features"], use_bn=True
    )

    # set the optimizer
    optimizer = optim.Adam(
        net_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # set the device
    train_nodes_features = train_nodes_features.to(device)
    # train_nodes_features, validation_nodes_features, test_nodes_features, labels = train_nodes_features.to(
    #     device), validation_nodes_features.to(device), test_nodes_features.to(device), labels.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    validation_labels = validation_labels.to(device)
    graph = graph.to(device)
    net_model = net_model.to(device)

    print("GCN Baseline")

    # start to train
    for epoch in range(200):
        # train
        # call the train method
        train(
            net_model,
            train_nodes_features,
            train_hyper_edge_list,
            graph,
            train_labels,
            train_edge_mask,
            optimizer,
            epoch,
        )

        if epoch % 1 == 0:
            with torch.no_grad():
                # validation(net_model, validation_nodes_features, validation_hyper_edge_list, graph_validation, labels, val_edge_mask)
                validation(
                    net_model,
                    train_nodes_features,
                    validation_hyper_edge_list,
                    graph,
                    validation_labels,
                    val_edge_mask,
                )

    test(
        net_model,
        train_nodes_features,
        test_hyper_edge_list,
        graph,
        test_labels,
        test_edge_mask,
    )

    # torch.save(net_model.state_dict(), "gcn_link.pkl")


if __name__ == "__main__":
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    main("Disease", "input link prediction dataset")
    # main("Disease", "output link prediction dataset")
    #
    # main("Immune System", "input link prediction dataset")
    # main("Immune System", "output link prediction dataset")
    #
    # main("Metabolism", "input link prediction dataset")
    # main("Metabolism", "output link prediction dataset")
    #
    # main("Signal Transduction", "input link prediction dataset")
    # main("Signal Transduction", "output link prediction dataset")
