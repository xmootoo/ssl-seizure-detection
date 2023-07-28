import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


import torch
from torch.utils.data import Dataset, DataLoader

class GraphPairDataset(Dataset):
    def __init__(self, edge_index, node_features, edge_features, labels):
        """
        Custom Dataset for handling paired graph data with node and edge features.

        Args:
            edge_index (list of tuples): List of numpy array tuples. Each tuple entry is a numpy array,
                                         where the first entry has shape (2, num_edges1), and the second entry of
                                         the tuple has shape (2, num_edges2), where num_edges1 and num_edges2 are
                                         the number of edges in the graph, and a column [i j]^T in any of the arrays
                                         represents the directed edges i -> j.
            node_features (list of tuples): List of numpy array tuples. Each tuple entry is of shape (num_nodes, num_node_features).
            edge_features (list of tuples): List of numpy array tuples. Each tuple entry is of shape (num_edges[i], num_node_features),
                                            where num_edges[i] means that it varies for every tuple entry.
            labels (list): List of corresponding labels for each graph pair.

        Note:
            Make sure that the edge_index, node_features, edge_features, and labels are aligned,
            such that the ith element of each list corresponds to the same graph pair and its features.

        Example:
            edge_index = [(edge_index_0_graph1, edge_index_1_graph1), (edge_index_0_graph2, edge_index_1_graph2), ...]
            node_features = [(node_features_graph1, node_features_graph1), (node_features_graph2, node_features_graph2), ...]
            edge_features = [(edge_features_graph1, edge_features_graph1), (edge_features_graph2, edge_features_graph2), ...]
            labels = [0, 1, 0, 1, ...]  # List of corresponding labels
            graph_pair_dataset = GraphPairDataset(edge_index, node_features, edge_features, labels)
        """

        self.edge_index = edge_index
        self.node_features = node_features
        self.edge_features = edge_features
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of paired graphs in the dataset.

        Returns:
            int: Total number of paired graphs in the dataset.
        """
        return len(self.edge_index)

    def __getitem__(self, index):
        """
        Retrieves a single paired graph and its features along with the corresponding label.

        Args:
            index (int): Index of the paired graph to retrieve.

        Returns:
            tuple: A tuple of three elements containing graph features for the first graph in the pair.
                The first element contains node features (NF) for the first graph,
                the second element contains edge index (A) for the first graph,
                and the third element contains edge features (EF) for the first graph.
            tuple: A tuple of three elements containing graph features for the second graph in the pair.
                The first element contains node features (NF) for the second graph,
                the second element contains edge index (A) for the second graph,
                and the third element contains edge features (EF) for the second graph.
            int: The corresponding label for the graph pair.
        """
        edge_index1, edge_index2 = self.edge_index[index]
        NF1, NF2 = self.node_features[index]
        EF1, EF2 = self.edge_features[index]
        Y = self.labels[index]

        return NF1, edge_index1, EF1, NF2, edge_index2, EF2, Y





def collate_fn(batch):
    """
    Custom collate function for processing batches of paired graph data with features.

    Args:
        batch (list): A list of tuples, where each tuple contains paired graph data and features, along with labels.

    Returns:
        tuple: A tuple of three elements representing the batched data.
            The first element contains batched node features (NF) for the first graph in the pair,
            the second element contains batched edge index (A) for the first graph in the pair,
            and the third element contains batched edge features (EF) for the first graph in the pair.
        tuple: A tuple of three elements representing the batched data.
            The first element contains batched node features (NF) for the second graph in the pair,
            the second element contains batched edge index (A) for the second graph in the pair,
            and the third element contains batched edge features (EF) for the second graph in the pair.
        tensor: A tensor containing batched labels for the graph pairs.
    """
    # Unzip the batch into separate lists for the first graph data, second graph data, and labels
    node_features1, edge_index1, edge_features1, node_features2, edge_index2, edge_features2, labels = zip(*batch)

    # Stack the elements to create batched tensors
    node_features1_batch = torch.stack([torch.tensor(nf) for nf in node_features1])
    edge_index1_batch = torch.cat([torch.tensor(ei) for ei in edge_index1], dim = 0)
    edge_features1_batch = torch.cat([torch.tensor(ef) for ef in edge_features1], dim = 0)

    node_features2_batch = torch.stack([torch.tensor(nf) for nf in node_features2])
    edge_index2_batch = torch.cat([torch.tensor(ei) for ei in edge_index2], dim = 0)
    edge_features2_batch = torch.cat([torch.tensor(ef) for ef in edge_features2], dim = 0)

    labels_batch = torch.tensor(labels)

    return (node_features1_batch, edge_index1_batch, edge_features1_batch), \
           (node_features2_batch, edge_index2_batch, edge_features2_batch), labels_batch



def rp_dataloader(edge_index, node_features, edge_features, labels, batch_size=32, shuffle=True):
    dataset = GraphPairDataset(edge_index, node_features, edge_features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader







# Test case for the script

# Example graphs represented by edge_index (list of numpy array tuples)
edge_index = [
    (np.array([[0, 0, 1, 2], [1, 2, 2, 3]]), np.array([[0, 1], [1, 2]])),
    (np.array([[0, 1], [1, 2]]), np.array([[0, 1], [1, 2]])),
    (np.array([[0, 0, 1], [1, 2, 2]]), np.array([[0, 1], [1, 2]]))
]

# Example node features (tuple of numpy arrays) for each graph in the dataset
node_features = [
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])),
    (np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]), np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])),
    (np.array([[20, 21, 22], [23, 24, 25], [26, 27, 28]]), np.array([[29, 30, 31], [32, 33, 34], [35, 36, 37]]))
]

# Example edge features (tuple of numpy arrays) for each graph in the dataset
edge_features = [
    (np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), np.array([[0.7, 0.8], [0.9, 1.0]])),
    (np.array([[1.1, 1.2], [1.3, 1.4]]), np.array([[1.5, 1.6], [1.7, 1.8]])),
    (np.array([[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]]), np.array([[2.7, 2.8], [2.9, 3.0]]))
]

# Example labels for each graph pair
labels = [0, 1, 0]


dataset = GraphPairDataset(edge_index, node_features, edge_features, labels)


print(dataset[0])


# # Creating DataLoader with the test case data
# data_loader = rp_dataloader(edge_index, node_features, edge_features, labels, batch_size=2, shuffle=True)

# # Iterating through the DataLoader to see the batches
# for batch_data1, batch_data2, batch_labels in data_loader:
#     print(batch_data1[0])
#     print(batch_data1[1])
#     print(batch_data2[0])
#     print(batch_data2[1])
    