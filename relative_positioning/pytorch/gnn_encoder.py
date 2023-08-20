# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
    
    
# Graph neural network encoder
class GNN_encoder(nn.Module):
    """
    Graph neural network embedder (encoder).

    Args:
        num_nodes (int): Number of nodes in the graph.
        nf_dim (list[int]): Node feature dimensions. For i > 0, nf_dim[i-1] is the input dimension of the i-th layer, and nf_dim[i] is the output dimension of the i-th layer.
        ef_dim (int): Number of input edge features.
        GAT_dim (int): Number of hidden units in the GAT layer.
        final_dim (int): Number of output features.
    """
    def __init__(self, num_nodes, nf_dim, ef_dim, num_heads, GAT_dim, hidden_dim, final_dim):
        super(GNN_encoder, self).__init__()

        #MLP 1, hidden layer with 32 units
        mlp1 = nn.Sequential(nn.Linear(ef_dim, 32), 
                             nn.ReLU(),
                             nn.Linear(32, nf_dim[0] * nf_dim[1]))
        
        # ECC layer
        self.ECC = NNConv(nf_dim[0], nf_dim[1], mlp1)

        # GAT layer
        self.GAT = GATConv(nf_dim[1], GAT_dim, heads=num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(num_nodes * GAT_dim * num_heads, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, final_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.

        Args:
            x (torch.Tensor): The node features of the graph. Shape: (num_nodes, num_node_features), where nf_dim[0] == num_node_features.
            edge_index (torch.Tensor): Edges indices of the graph for the edge features. Shape: (2, num_edges) where num_edges is the number of edges in the graph. 
                                       The first row contains the source node indices and the second row contains the target node indices. For example, the column [4 2]^T refers to the directed edge from
                                       node 4 to node 2.  
            edge_attr (torch.Tensor): Edge features of the graph. Shape: (num_edges, ef_dim), where ef_dim is the number of input edge features. Each row in the tensor corresponds to the edge-specific feature,
                                        i.e. the edge feature of the edge specified by the corresponding column in edge_index.

        Returns:
            x (torch.Tensor): The graph embedding vector. Shape: (final_dim,).
        """
        # ECC Convolution
        x = F.relu(self.ECC(x, edge_index, edge_attr))

        # GAT Convolution
        x = F.relu(self.GAT(x, edge_index))

        # Flatten the features
        x = x.view(-1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        return x



# Test case for the GNN_encoder
num_nodes = 40
num_node_features = 5
num_edges = 20
num_edge_features = 2
nf_dim = [num_node_features, 64]  # Node feature dimensions for each layer
ef_dim = num_edge_features  # Edge feature dimension
num_heads = 1  # Number of GAT heads
GAT_dim = 27  # Number of hidden units in the GAT layer
hidden_dim = 450  # Number of hidden units in first fully connected layer
final_dim = 128  # Number of output features

# Create random node features and edge features
x = torch.randn(num_nodes, num_node_features)
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
edge_attr = torch.randn(num_edges, num_edge_features)

# Create an instance of the GNN_encoder
encoder = GNN_encoder(num_nodes, nf_dim, ef_dim, num_heads, GAT_dim, hidden_dim, final_dim)

# Forward pass through the encoder
output = encoder(x, edge_index, edge_attr)
print("Output shape:", output.shape)