import torch
from sklearn.metrics import ndcg_score

from utils import (
    read_out_to_generate_single_hyper_edge_embedding,
    read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list,
    encode_edges_features,
)

if __name__ == "__main__":
    list_of_nodes_for_single_hyper_edge = [[0, 1]]

    # nodes_features = torch.Tensor([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # test = read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(list_of_nodes_for_single_hyper_edge, nodes_features)
    # test = encode_edges_features(list_of_nodes_for_single_hyper_edge, 2, 5)
    outs = [[1, 0.9, 0.2, 0.9], [0.1, 0.2, 0.3, 0.9]]
    labels = [[0, 1, 1, 1], [0, 0, 0, 1]]
    ndcg_res = ndcg_score(labels, outs)

    print(ndcg_res)
