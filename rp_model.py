# Libraries
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import NNConv, GATConv



# Graph neural network encoder
class GNN_enc(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_features, num_heads, hidden_dim):
        super(GNN_enc, self).__init__()

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

print("done")