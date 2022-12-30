import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.models import GCN
from sklearn.metrics import ndcg_score

from data_loader import DataLoaderAttribute

learning_rate = 0.01
weight_decay = 5e-4


def train(
    net_model: torch.nn.Module,
    nodes_features: torch.Tensor,
    graph: Graph,
    labels: torch.Tensor,
    train_idx: list[bool],
    optimizer: optim.Adam,
    epoch: int,
):
    net_model.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net_model(nodes_features, graph)

    outs = outs[train_idx]
    loss = F.cross_entropy(outs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def validation(net_model, nodes_features, graph, labels, validation_idx):
    net_model.eval()

    outs = net_model(nodes_features, graph)

    outs = outs[validation_idx]

    val_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())

    print(
        "\033[1;32m"
        + "The validation score is: "
        + "{:.5f}".format(val_res)
        + "\033[0m"
    )


@torch.no_grad()
def test(net_model, nodes_features, graph, labels, test_idx):
    net_model.eval()

    outs = net_model(nodes_features, graph)

    outs = outs[test_idx]

    test_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())

    print(
        "\n\033[1;35m"
        + "The final test score is: "
        + "{:.5f}".format(test_res)
        + "\033[0m"
    )


if __name__ == "__main__":
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # initialize the data_loader
    data_loader = DataLoaderAttribute("Disease", "attribute prediction dataset")

    # get the labels - the original nodes features
    # labels = torch.FloatTensor(data_loader["raw_nodes_features"])
    train_labels = data_loader["train_labels"]
    validation_labels = data_loader["validation_labels"]
    test_labels = data_loader["test_labels"]

    # get the train,val,test nodes features
    train_nodes_features = torch.FloatTensor(data_loader["train_nodes_features"])
    validation_nodes_features = torch.FloatTensor(
        data_loader["validation_nodes_features"]
    )
    test_nodes_features = torch.FloatTensor(data_loader["test_nodes_features"])

    # get train, validation, test mask to track the nodes
    train_mask = data_loader["train_node_mask"]
    val_mask = data_loader["val_node_mask"]
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

    # the GCN model
    net_model = GCN(
        data_loader["num_features"], 32, data_loader["num_features"], use_bn=True
    )

    # set the optimizer
    optimizer = optim.Adam(
        net_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # set the device
    train_nodes_features, validation_nodes_features, test_nodes_features = (
        train_nodes_features.to(device),
        validation_nodes_features.to(device),
        test_nodes_features.to(device),
    )
    train_labels = train_labels.to(device)
    validation_labels = validation_labels.to(device)
    test_labels = test_labels.to(device)

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
            graph,
            train_labels,
            train_mask,
            optimizer,
            epoch,
        )

        if epoch % 1 == 0:
            with torch.no_grad():
                validation(
                    net_model,
                    validation_nodes_features,
                    graph,
                    validation_labels,
                    val_mask,
                )

    test(net_model, test_nodes_features, graph, test_labels, test_mask)
    # torch.save(net_model.state_dict(), "gcn.pkl")
