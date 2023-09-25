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

#TODO: Implement this into the relative_positioning class along with a projector.
class gnn_encoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels):
        super(gnn_encoder, self).__init__()
        
        #MLP for NNConv (e.g., dynamic filter-generating network)
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels)
        
        # NNConv layer
        self.conv1 = NNConv(num_node_features, hidden_channels, self.edge_mlp)
        
        # GATConv layer
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, 1)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # NNConv layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # GATConv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # NON-global average pooling

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        
        return x
        
        



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
    
        
    def encoder(self, x, edge_index, edge_attr, batch):
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
        z1 = self.encoder(x1, edge_index1, edge_attr1, batch1)
        
        # Second graph's embeddings
        z2 = self.encoder(x2, edge_index2, edge_attr2, batch2)
        
        # Contrast the embeddings
        z = torch.abs(z1 - z2)
        
        # Logistic regression
        z = self.fc2(z)
        
        if mode == "sigmoid":
            z = torch.sigmoid(z)
            
        elif mode == "linear":
            pass
        
        return z.squeeze(1)
    

class temporal_shuffling(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels):
        super(temporal_shuffling, self).__init__()
        
        # Initialize the MLP for NNConv
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels)
        
        # NNConv layer
        self.conv1 = NNConv(num_node_features, hidden_channels, self.edge_mlp)
        
        # GATConv layer
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        self.fc2 = nn.Linear(2 * out_channels, 1)
    
        
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
    
    def forward(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2, x3, edge_index3, edge_attr3, batch3, mode="sigmoid"):
        # First graph's embeddings
        z1 = self.embedder(x1, edge_index1, edge_attr1, batch1)
        
        # Second graph's embeddings
        z2 = self.embedder(x2, edge_index2, edge_attr2, batch2)
        
        # Third graph's embeddings
        z3 = self.embedder(x3, edge_index3, edge_attr3, batch3)
        
        # Contrast the embeddings
        diff1 = torch.abs(z1 - z2)
        diff2 = torch.abs(z2 - z3)
        z = torch.cat((diff1, diff2), dim=1)

        
        # Logistic regression
        z = self.fc2(z)
        
        if mode == "sigmoid":
            z = torch.sigmoid(z)
            
        elif mode == "linear":
            pass
        
        return z.squeeze(1)
    

class supervised_model(nn.Module):
    def __init__(self, ):
        super(supervised_model, self).__init__()
        
