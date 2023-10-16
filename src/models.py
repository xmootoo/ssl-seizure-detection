# Model
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
from torch_geometric.graphgym.init import init_weights
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the MLP for NNConv
class EdgeMLP(nn.Module):
    def __init__(self, num_edge_features, in_channels, out_channels):
        super(EdgeMLP, self).__init__()
        
        # Define a sequential architecture
        self.mlp = nn.Sequential(
            nn.Linear(num_edge_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * out_channels)
        )
    
        # Weight initialization
        self.apply(init_weights)
    
    def forward(self, edge_attr):
        x = self.mlp(edge_attr)
        return x


class gnn_encoder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(gnn_encoder, self).__init__()
        
        # Initialize the MLP for NNConv
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels[0])
        
        # Encoder
        self.conv1 = NNConv(num_node_features, hidden_channels[0], self.edge_mlp)
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1], heads=1, concat=False)

        # Projector
        self.fc1 = nn.Linear(hidden_channels[1], hidden_channels[2])
        self.fc2 = nn.Linear(hidden_channels[2], hidden_channels[3])
        self.fc3 = nn.Linear(hidden_channels[3], hidden_channels[4])
        
        # Weight initialization
        self.apply(init_weights)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # NNConv layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        # GATConv layer
        x = F.relu(self.conv2(x, edge_index))

        # Global average pooling
        x = global_mean_pool(x, batch) #<-- batch vector to keep track of graphs

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x
        


class relative_positioning(nn.Module):
    def __init__(self, config):
        super(relative_positioning, self).__init__()
        num_node_features = config["num_node_features"]
        num_edge_features = config["num_edge_features"]
        hidden_channels = config["hidden_channels"]
        
        # GNN encoder
        self.encoder = gnn_encoder(num_node_features, num_edge_features, hidden_channels)

        # Fully connected layers
        self.fc = nn.Linear(hidden_channels[3], 1)
        
        # Weight initialization
        self.apply(init_weights)
    
    
    def forward(self, batch, head="linear"):
        # Graph embeddings
        z1 = self.encoder(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch)
        z2 = self.encoder(batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
        
        # Contrast the embeddings
        z = torch.abs(z1 - z2)
        
        # Linear or Logistic regression
        z = self.fc(z)
        
        
        if head == "sigmoid":
            z = torch.sigmoid(z)
            
        elif head == "linear":
            pass
        
        return z.squeeze(1)


    

class temporal_shuffling(nn.Module):
    def __init__(self, config):
        super(temporal_shuffling, self).__init__()
        num_node_features = config["num_node_features"]
        num_edge_features = config["num_edge_features"]
        hidden_channels = config["hidden_channels"]
        
        # GNN Encoder
        self.encoder = gnn_encoder(num_node_features, num_edge_features, hidden_channels)
        
        # Fully connected layer
        self.fc = nn.Linear(2 * hidden_channels[3], 1)
        
        # Weight initialization
        self.apply(init_weights)


    def forward(self, batch, head="linear"):
        # Encoding for each graph
        z1 = self.encoder(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch)
        z2 = self.encoder(batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
        z3 = self.encoder(batch.x3, batch.edge_index3, batch.edge_attr3, batch.x3_batch)
        
        # Contrast the embeddings
        diff1 = torch.abs(z1 - z2)
        diff2 = torch.abs(z2 - z3)
        z = torch.cat((diff1, diff2), dim=1)

        # Logistic regression
        z = self.fc(z)
            
        if head == "linear":
            pass
        elif head == "sigmoid":
            z = torch.sigmoid(z)

        return z.squeeze(1)
    


class supervised_model(nn.Module):
    def __init__(self, config):
        super(supervised_model, self).__init__()
        num_node_features = config["num_node_features"]
        num_edge_features = config["num_edge_features"]
        hidden_channels = config["hidden_channels"]
        out_channels = config["out_channels"]
        dropout = config["dropout"]
        
        # Initialize the MLP for NNConv
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels)
        
        # NNConv layer
        self.conv1 = NNConv(num_node_features, hidden_channels, self.edge_mlp)
        
        # GATConv layer
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)

        # First fully connected layer
        self.fc1 = nn.Linear(hidden_channels, out_channels)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Last fully connected layer
        self.fc2 = nn.Linear(out_channels, 1)
        self.fc3 = nn.Linear(out_channels, 3)
        
        # Weight initialization
        self.apply(init_weights)
    
    def forward(self, batch, classify="binary", head="linear", dropout=True):

        # ECC
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)

        # GAT
        x = self.conv2(x, batch.edge_index)
        x = F.relu(x)

        # Global average pooling
        x = global_mean_pool(x, batch.batch)


        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        if dropout:
            x = self.dropout(x)
        
        
        if classify == "binary":
            x = self.fc2(x)
            x = x.squeeze(1)
        if classify == "multiclass":
            x = self.fc3(x)
        
        if head == "linear":
            return x
        if head == "sigmoid":
            return torch.sigmoid(x)
        if head == "softmax":
            return torch.softmax(x, dim=1)

        

        
class CPC(nn.Module):
    def __init__(self):
        super(CPC, self).__init__()
    

class VICReg(nn.Module):
    def __init__(self):
        super(VICReg, self).__init__()