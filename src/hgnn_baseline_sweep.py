import pprint
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dhg import Hypergraph
from dhg.models import HGNN
from sklearn.metrics import accuracy_score, ndcg_score

from data_loader import DataLoaderAttribute

model_name = "HGNN"
project_name = "pathway_attribute_predict"


def train(
    net_model: torch.nn.Module,
    nodes_features: torch.Tensor,
    graph: Hypergraph,
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
    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)
    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)
    print(
        "\033[1;32m" + "The validate ndcg is: " + "{:.5f}".format(ndcg_res) + "\033[0m"
    )
    print(
        "\033[1;32m"
        + "The validate accuracy is: "
        + "{:.5f}".format(acc_res)
        + "\033[0m"
    )
    return ndcg_res, acc_res


@torch.no_grad()
def test(net_model, nodes_features, graph, labels, test_idx):
    net_model.eval()
    outs = net_model(nodes_features, graph)
    outs = outs[test_idx]
    cat_labels = labels.cpu().numpy().argmax(axis=1)
    cat_outs = outs.cpu().numpy().argmax(axis=1)
    ndcg_res = ndcg_score(labels.cpu().numpy(), outs.cpu().numpy())
    acc_res = accuracy_score(cat_labels, cat_outs)
    print("\033[1;32m" + "The test ndcg is: " + "{:.5f}".format(ndcg_res) + "\033[0m")
    print(
        "\033[1;32m" + "The test accuracy is: " + "{:.5f}".format(acc_res) + "\033[0m"
    )
    return ndcg_res, acc_res

# if __name__ == "__main__":
def main():
    with wandb.init(project=project_name):
        config = wandb.config
        print(config)
        # set device
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

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

        # the HGNN model
        net_model = HGNN(
            data_loader["num_features"],
            config.emb_dim,
            data_loader["num_features"],
            use_bn=True,
            drop_rate=config.drop_out,
        )

        # set the optimizer
        optimizer = optim.Adam(
            net_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
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

        print("HGNN Baseline")

        # start to train
        for epoch in range(200):
            # train
            # call the train method
            loss = train(
                net_model,
                train_nodes_features,
                hyper_graph,
                train_labels,
                train_mask,
                optimizer,
                epoch,
            )

            if epoch % 1 == 0:
                with torch.no_grad():
                    valid_ndcg, valid_acc = validation(
                        net_model,
                        validation_nodes_features,
                        hyper_graph,
                        validation_labels,
                        val_mask,
                    )
                    test_ndcg, test_acc = test(
                        net_model, test_nodes_features, hyper_graph, test_labels, test_mask
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


task = "attribute prediction dataset"
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
    sweep_id = wandb.sweep(sweep_config, project="pathway_attribute_predict_sweep")

    wandb.agent(sweep_id, main)
