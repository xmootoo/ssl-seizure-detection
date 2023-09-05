# Model
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the MLP for NNConv
class EdgeMLP(nn.Module):
    def __init__(self, num_edge_features, in_channels, out_channels):
        super(EdgeMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_edge_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * out_channels)
        )
        
    def forward(self, edge_attr):
        return self.mlp(edge_attr)


# Adapted Graph Neural Network using NNConv and GATConv
class relative_positioning(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels):
        super(relative_positioning, self).__init__()
        
        # Initialize the MLP for NNConv
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels)
        
        # NNConv layer
        self.conv1 = NNConv(num_node_features, hidden_channels, self.edge_mlp)
        
        # GATConv layer
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, 1)
    
        
    def embedder(self, x, edge_index, edge_attr, batch):
        # NNConv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # GATConv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global average pooling
        x = global_mean_pool(x, batch) #<-- batch vector to keep track of graphs

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        
        return x
    
    def forward(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2, mode="sigmoid"):
        # First graph's embeddings
        z1 = self.embedder(x1, edge_index1, edge_attr1, batch1)
        
        # Second graph's embeddings
        z2 = self.embedder(x2, edge_index2, edge_attr2, batch2)
        
        # Contrast the embeddings
        z = torch.abs(z1 - z2)
        
        # Logistic regression
        z = self.fc2(z)
        
        if mode == "sigmoid":
            z = torch.sigmoid(z)
            
        elif mode == "linear":
            pass
        
        return z.squeeze(1)