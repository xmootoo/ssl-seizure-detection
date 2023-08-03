# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GATConv, global_mean_pool



# Graph neural network encoder
# Graph neural network encoder
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
    def __init__(self, num_nodes, nf_dim, ef_dim, num_heads, GAT_dim, final_dim):
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
        self.fc = nn.Linear(GAT_dim, final_dim)

    def forward(self, data):
        """
        Forward pass.

        Args:
            node_features (torch.Tensor): The node features of the graph. Shape: (num_nodes, num_node_features).
            edge_index (torch.Tensor): Edges indices of the graph. Shape: (2, num_edges) where num_edges is the number of edges in the graph. 
                                       The first row contains the source node indices and the second row contains the target node indices. For example, the column [4 2]^T refers to the directed edge from
                                       node 4 to node 2.  
            edge_features (torch.Tensor): Edge features of the graph. Shape: (num_edges, ef_dim), where ef_dim is the number of input edge features. Each row in the tensor corresponds to the edge-specific feature,
                                        indexed by the corresponding column in edge_index.

        Returns:
            x (torch.Tensor): The graph embedding vector. Shape: (final_dim,).
        """
        # Obtain the node features, edge indices, and edge features
        x, edge_index, edge_attr = data.x.float(), data.edge_index.long(), data.edge_attr.float()
        
        # ECC Convolution
        x = F.relu(self.ECC(x, edge_index, edge_attr))

        # GAT Convolution
        x = F.relu(self.GAT(x, edge_index))

        # Global average pooling
        x = global_mean_pool(x, data.batch)

        # Create a new tensor to hold the edge features with the right number of columns
        print(x.shape)

        # # Flatten the features
        # x = x.view(-1)
        
        print(x.shape)

        # Apply the fully connected layer
        x = F.relu(self.fc(x))
        
        return x


# Contrastive module
class Contrast(nn.Module):
    """
    The contrastive module to measure the difference of embeddings.

    Args:
        z_1 (tensor): Embedding of the first graph.
        z_2 (tensor): Embedding of the second graph.
    
    Returns:
        x (tensor): The absolute value (entrywise) difference of the embeddings.
    """
    
    def __init__(self):
        super(Contrast, self).__init__()

    def forward(self, z_1, z_2):
        x = torch.abs(z_1 - z_2)
        
        return x


# Logistic regression
class Regression(nn.Module):
    """
    Linear regression module. We will turn this into a logistic regression by applying a sigmoid function to the output with
    the BCEWithLogisticLoss function (which applies the sigmoid function before taking the standard BCELoss).

    Args:
        in_dim (int): Number of input features.
        
    Returns:
        x (float): The probability of the input belonging to the positive class.
        
    """
    
    def __init__(self, in_dim):
        super(Regression, self).__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.fc(x.float())
        
        return x