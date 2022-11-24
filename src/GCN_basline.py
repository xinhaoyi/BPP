import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Graph
from dhg import Graph, Hypergraph
from dhg.data import Cooking200
from dhg.models import GCN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator


def train(net_model: torch.nn.Module, nodes_features: torch.Tensor, graph: Graph, labels: torch.Tensor,
          train_idx: list[bool],
          optimizer: optim.Adam, epoch: int):
    net_model.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net_model(nodes_features, graph)
    # outs: torch.Size([7403, 2000]) 就是7403个点，每个点有 2000个 features
    # labels: torch.Size([7403, 2000]) 就是7403个点，和每个点的真实的 2000个 features
    outs, labels = outs[train_idx], labels[train_idx]
    loss = F.cross_entropy(outs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, nodes_features, graph, labels, idx, evaluator, test=False):
    net.eval()
    outs = net(nodes_features, graph)
    outs, labels = outs[idx], labels[idx]
    if not test:
        res = evaluator.validate(labels, outs)
    else:
        res = evaluator.test(labels, outs)
    return res

if __name__ == '__main__':
    test = []
    sum(test)


