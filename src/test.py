import torch

from utils import read_out_to_generate_single_hyper_edge_embedding, \
    read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list, encode_edges_features

if __name__ == '__main__':
    list_of_nodes_for_single_hyper_edge = torch.Tensor([[0, 1, 3], [0, 1, 2]])
    nodes_features = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]])
    test = read_out_to_generate_multi_hyper_edges_embeddings_from_edge_list(list_of_nodes_for_single_hyper_edge, nodes_features)
    # test = encode_edges_features(list_of_nodes_for_single_hyper_edge, 2, 5)

    print(test)
