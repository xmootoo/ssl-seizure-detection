import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


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

        # Convert numpy arrays to PyTorch tensors
        edge_index1 = torch.from_numpy(edge_index1)
        edge_index2 = torch.from_numpy(edge_index2)
        NF1 = torch.from_numpy(NF1)
        NF2 = torch.from_numpy(NF2)
        EF1 = torch.from_numpy(EF1)
        EF2 = torch.from_numpy(EF2)
        Y = torch.tensor(Y)

        # Create Data objects for each graph
        data1 = Data(x=NF1, edge_index=edge_index1, edge_attr=EF1)
        data2 = Data(x=NF2, edge_index=edge_index2, edge_attr=EF2)

        return data1, data2, Y



def collate_fn(batch):
    """
    Custom collate function for processing batches of paired graph data with features.
    This function converts a batch of graph pairs into a PyTorch Geometric Batch object and a tensor of labels.
    The Batch object automatically adjusts edge indices when concatenating multiple graphs.

    Args:
        batch (list): A list of tuples, where each tuple contains paired graph data and features, along with labels.
            Each tuple contains:
            - node features of the first graph (numpy array of shape (num_nodes1, num_node_features))
            - edge indices of the first graph (numpy array of shape (2, num_edges1))
            - edge features of the first graph (numpy array of shape (num_edges1, num_edge_features))
            - node features of the second graph (numpy array of shape (num_nodes2, num_node_features))
            - edge indices of the second graph (numpy array of shape (2, num_edges2))
            - edge features of the second graph (numpy array of shape (num_edges2, num_edge_features))
            - label of the graph pair (int)

    Returns:
        Batch: A PyTorch Geometric Batch object containing the batched data. The Batch object contains the following fields:
            - batch.x: Node feature matrix (torch tensor of shape (total_num_nodes, num_node_features))
            - batch.edge_index: Edge index matrix (torch tensor of shape (2, total_num_edges))
            - batch.edge_attr: Edge feature matrix (torch tensor of shape (total_num_edges, num_edge_features))
            Here, total_num_nodes is the total number of nodes in all graphs in the batch, and total_num_edges is the total number of edges.
        tensor: A torch tensor containing batched labels for the graph pairs (shape: (batch_size,))
    """
    data_list = []
    labels = []
    for data1, data2, label in batch:
        data_list.append(data1)
        data_list.append(data2)
        labels.append(label)
    batch = Batch.from_data_list(data_list)
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    return batch, labels



def rp_dataloader(edge_index, node_features, edge_features, labels, batch_size=32, shuffle=True):
    """
    Creates a DataLoader for paired graph data.

    Args:
        edge_index (list of tuples): List of edge indices.
        node_features (list of tuples): List of node features.
        edge_features (list of tuples): List of edge features.
        labels (list of): List of labels.
        batch_size (int, optional): Number of graph pairs per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.

    Returns:
        DataLoader: A DataLoader object for the graph pair data.
    """
    dataset = GraphPairDataset(edge_index, node_features, edge_features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader








# Test case
num_graphs = 5
num_node_features = 3
num_edge_features = 4

edge_index = []
node_features = []
edge_features = []
for _ in range(num_graphs):
    num_nodes1, num_nodes2 = 8, 8
    num_edges1, num_edges2 = np.random.randint(5, 8, size=2)
    edge_index.append((np.random.randint(num_nodes1, size=(2, num_edges1)), np.random.randint(num_nodes2, size=(2, num_edges2))))
    node_features.append((np.random.rand(num_nodes1, num_node_features), np.random.rand(num_nodes2, num_node_features)))
    edge_features.append((np.random.rand(num_edges1, num_edge_features), np.random.rand(num_edges2, num_edge_features)))

labels = np.random.randint(2, size=num_graphs)

# Create the DataLoader
data_loader = rp_dataloader(edge_index, node_features, edge_features, labels, batch_size=2, shuffle=False)


# Assume data_loader is your DataLoader object
for batch_idx, (batch, batch_labels) in enumerate(data_loader):
    print(f"Processing batch {batch_idx}...")
    
    # You can access the batch data here
    for i in range(0, batch.num_graphs, 2):  # Step by 2 because each pair contains 2 graphs
        # Get the nodes for the first graph in the pair
        nodes1 = batch.x[batch.batch == i]
        edge_mask1 = batch.batch[batch.edge_index[0].long()] == i
        edges1 = batch.edge_index[:, edge_mask1]
        edge_attr1 = batch.edge_attr[edge_mask1]

        # Get the nodes for the second graph in the pair
        nodes2 = batch.x[batch.batch == i + 1]
        edge_mask2 = batch.batch[batch.edge_index[0].long()] == i + 1
        edges2 = batch.edge_index[:, edge_mask2]
        edge_attr2 = batch.edge_attr[edge_mask2]

        
        # edge_mask1 is a boolean mask for the edges of the first graph in a pair. It is used to separate the edges of the 
        # first graph from the edges of the second graph in the edge_index tensor of the batch. It is created by checking 
        # which edges (represented by the source nodes in the edge_index tensor) belong to the first graph based on the 
        # batch assignment of nodes. Similarly for edge_mask2.
        
        # Now nodes1, edges1, and edge_attr1 contain the node features, edge indices, and edge features for the first graph in the pair,
        # and nodes2, edges2, and edge_attr2 contain the node features, edge indices, and edge features for the second graph in the pair

        print(f"Graph pair {i // 2} in batch {batch_idx}:")
        print(f"First graph nodes: {nodes1}")
        print(f"First graph edges: {edges1}")
        print(f"First graph edge attributes: {edge_attr1}")
        print(f"Second graph nodes: {nodes2}")
        print(f"Second graph edges: {edges2}")
        print(f"Second graph edge attributes: {edge_attr2}")

