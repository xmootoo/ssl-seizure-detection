# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GATConv



# Graph neural network encoder
class GNN_embedder(nn.Module):
    """
    Graph neural network embedder (encoder).

    Args:
        in_channels (int): Number of input features?
    """
    def __init__(self, in_channels, out_channels, num_edge_features, num_heads, hidden_dim):
        super(GNN_embedder, self).__init__()

        # ECC layer
        self.ecc_conv = NNConv(in_channels, out_channels, num_edge_features)

        # GAT layer
        self.gat_conv = GATConv(out_channels, hidden_dim, heads=num_heads)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, x, edge_index, edge_attr):
        # ECC Convolution
        x = F.relu(self.ecc_conv(x, edge_index, edge_attr))

        # GAT Convolution
        x = F.relu(self.gat_conv(x, edge_index))

        # Flatten the features
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

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
class LogisticRegression(nn.Module):
    """
    The logistic regression module.

    Args:
        
    Returns:
        
    """
    
    def __init__(self, in_dim):
        super(LogisticRegression).__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        
        return x
    
print("done")