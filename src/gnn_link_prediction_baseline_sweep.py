import copy
import pprint
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dhg import Graph, Hypergraph
from dhg.models import GCN, HGNN, HGNNP
from sklearn.metrics import accuracy_score, ndcg_score

import utils
from data_loader import DataLoaderLink

learning_rate = 0.01
weight_decay = 5e-4
project_name = "gnn_link_prediction_sweep"

def train(
    net_model: torch.nn.Module,
    nodes_features: torch.Tensor,
    train_hyper_edge_list: list[list[int]],
    graph,
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

    edges_embeddings = edges_embeddings.to(net_model.device)

    nodes_embeddings = net_model(nodes_features, graph)

    # torch.backends.cudnn.enabled = False
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

    edges_embeddings = edges_embeddings.to(net_model.device)
    nodes_embeddings = net_model(nodes_features, graph)

    # torch.backends.cudnn.enabled = False
    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)
    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)
    print(
        "\033[1;32m"
        + "The validation ndcg is: "
        + "{:.5f}".format(ndcg_res)
        + "\033[0m"
    )
    print(
        "\033[1;32m"
        + "The validation accuracy is: "
        + "{:.5f}".format(acc_res)
        + "\033[0m"
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
    # [[1,2,3],[2,3,4,5]...]
    edges_embeddings = (
        utils.read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(
            test_hyper_edge_list, nodes_features
        )
    )

    edges_embeddings = edges_embeddings.to(net_model.device)

    nodes_embeddings = net_model(nodes_features, graph)

    outs = torch.matmul(edges_embeddings, nodes_embeddings.t())

    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)
    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)
    print("\033[1;32m" + "The test ndcg is: " + "{:.5f}".format(ndcg_res) + "\033[0m")
    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )
    return ndcg_res, acc_res


def main():
    with wandb.init(project=project_name):
        config = wandb.config
        print(config)
        # set device
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

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
        hyper_graph_train = Hypergraph(
            num_of_nodes, copy.deepcopy(train_hyper_edge_list)
        )

        # generate train graph based on hyper graph
        graph_train = Graph.from_hypergraph_clique(hyper_graph_train, weighted=True)

        # the train hyper graph
        hyper_graph_validation = Hypergraph(
            num_of_nodes, copy.deepcopy(train_hyper_edge_list)
        )

        # generate train graph based on hyper graph
        graph_validation = Graph.from_hypergraph_clique(
            hyper_graph_validation, weighted=True
        )

        # the train hyper graph
        hyper_graph_test = Hypergraph(
            num_of_nodes, copy.deepcopy(train_hyper_edge_list)
        )

        # generate train graph based on hyper graph
        graph_test = Graph.from_hypergraph_clique(hyper_graph_test, weighted=True)

        # the GCN model
        if config.model_name == "GCN":
            net_model = GCN(
                data_loader["num_features"],
                config.emb_dim,
                data_loader["num_features"],
                use_bn=True,
                drop_rate=config.drop_out,
            )
        elif config.model_name == "HGNN":
            # the HGNN model
            net_model = HGNN(
                data_loader["num_features"],
                config.emb_dim,
                data_loader["num_features"],
                use_bn=True,
                drop_rate=config.drop_out,
            )
        elif config.model_name == "HGNNP":
            # the HGNNP model
            net_model = HGNNP(
                data_loader["num_features"],
                config.emb_dim,
                data_loader["num_features"],
                use_bn=True,
                drop_rate=config.drop_out,
            )
        else:
            raise Exception("Sorry, no model_name has been recognized.")
        net_model.device = device
        # set the optimizer
        optimizer = optim.Adam(
            net_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # set the device
        train_nodes_features, train_labels, test_labels, validation_labels = (
            train_nodes_features.to(device),
            train_labels.to(device),
            test_labels.to(device),
            validation_labels.to(device),
        )
        if config.model_name != "GCN":
            graph_train = hyper_graph_train
            graph_validation = hyper_graph_validation
            graph_test = hyper_graph_test
        
        graph_train = graph_train.to(device)
        graph_validation = graph_validation.to(device)
        graph_test = graph_test.to(device)
        net_model = net_model.to(device)

        print(f"{config.model_name} Baseline")

        # start to train
        for epoch in range(200):
            # train
            # call the train method
            loss = train(
                net_model,
                train_nodes_features,
                train_hyper_edge_list,
                graph_train,
                train_labels,
                train_edge_mask,
                optimizer,
                epoch,
            )

            if epoch % 1 == 0:
                with torch.no_grad():
                    # validation(net_model, validation_nodes_features, validation_hyper_edge_list, graph_validation, labels, val_edge_mask)
                    valid_ndcg, valid_acc = validation(
                        net_model,
                        train_nodes_features,
                        validation_hyper_edge_list,
                        graph_validation,
                        validation_labels,
                        val_edge_mask,
                    )
                    test_ndcg, test_acc = test(
                        net_model,
                        train_nodes_features,
                        test_hyper_edge_list,
                        graph_test,
                        test_labels,
                        test_edge_mask,
                    )
                    wandb.log(
                        {
                            "loss": loss,
                            "epoch": epoch,
                            "valid_ndcg": valid_ndcg,
                            "valid_acc": valid_acc,
                            "test_ndcg": test_ndcg,
                            "test_acc": test_acc,
                        }
                    )


model_name = input()
print(f"start tunning {model_name}")
for task in ["output link prediction dataset", "input link prediction dataset"]:
    for dataset in ["Immune System", "Metabolism", "Signal Transduction", "Disease"]:
        sweep_config = {"method": "grid"}
        metric = {"name": "valid_ndcg", "goal": "maximize"}
        sweep_config["metric"] = metric
        parameters_dict = {
            "learning_rate": {"values": [0.05, 0.01, 0.005]},
            "emb_dim": {"values": [64, 128, 256]},
            "drop_out": {"values": [0.5, 0.6, 0.7]},
            "weight_decay": {"values": [5e-4]},
            "model_name": {"values": [model_name]},
            "task": {"values": [task]},
            "dataset": {"values": [dataset]},
        }
        sweep_config["parameters"] = parameters_dict
        pprint.pprint(sweep_config)
        sweep_id = wandb.sweep(sweep_config, project=f"{task}_sweep")
        wandb.agent(sweep_id, main)
