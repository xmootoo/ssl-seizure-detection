# Model
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
from torch_geometric.graphgym.init import init_weights
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeMLP(nn.Module):
    def __init__(self, num_edge_features, input_node_features, output_node_features):
        super(EdgeMLP, self).__init__()
        
        # Define a sequential architecture
        self.mlp = nn.Sequential(
            nn.Linear(num_edge_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_node_features * output_node_features)
        )
    
        # Weight initialization
        self.apply(init_weights)
    
    def forward(self, edge_attr):
        x = self.mlp(edge_attr)
        return x


class gnn_embedder(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(gnn_embedder, self).__init__()
        
        # Initialize the MLP for NNConv
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels[0])
        
        # embedder
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
        
        # GNN embedder
        self.embedder = gnn_embedder(num_node_features, num_edge_features, hidden_channels)

        # Fully connected layers
        self.fc = nn.Linear(hidden_channels[4], 1)
        
        # Weight initialization
        self.apply(init_weights)
    
    
    def forward(self, batch, head="linear"):
        # Graph embeddings
        z1 = self.embedder(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch)
        z2 = self.embedder(batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
        
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
        
        # GNN embedder
        self.embedder = gnn_embedder(num_node_features, num_edge_features, hidden_channels)
        
        # Fully connected layer
        self.fc = nn.Linear(2 * hidden_channels[4], 1)
        
        # Weight initialization
        self.apply(init_weights)


    def forward(self, batch, head="linear"):
        # embedding for each graph
        z1 = self.embedder(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch)
        z2 = self.embedder(batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
        z3 = self.embedder(batch.x3, batch.edge_index3, batch.edge_attr3, batch.x3_batch)
        
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


class downstream1(nn.Module):
    def __init__(self, config, pretrained_layers, frozen=False):
        super(downstream1, self).__init__()
        """
        Downstream model for seizure detection (binary or multiclass). Retrains with a GNN embedder and adds additional layers (total: 2x ECC, 2x GAT).
        
        Args:
            config (dict): Dictionary containing the configuration of the model, containing hidden_channels which is a list of values of length 3, and 
                            the dropout probability.
            pretrained_layers (tuple): Tuple containing the pretrained layers.
            frozen (bool): If True, the pretrained layers are frozen. If False, the pretrained layers are unfrozen.
        
        """
        
        
        hidden_channels = config["hidden_channels"]
        dropout = config["dropout"]
        
        # Pretrained layers
        EdgeMLP_pretrained, NNConv_pretrained, GATConv_pretrained = pretrained_layers
        self.conv1 = NNConv_pretrained
        self.conv2 = GATConv_pretrained

        # Assign the pretrained MLP to the NNConv1
        NNConv_pretrained.edge_mlp = EdgeMLP_pretrained
        
        # Output feature dimensions of pretrained layers
        num_node_features = GATConv_pretrained.out_channels
        num_edge_features = EdgeMLP_pretrained.state_dict()['mlp.0.weight'].size()[1]
        
        # Conditionally freeze or unfreeze pretrained layers
        for param in EdgeMLP_pretrained.parameters():
            param.requires_grad = not frozen
        for param in NNConv_pretrained.parameters():
            param.requires_grad = not frozen
        for param in GATConv_pretrained.parameters():
            param.requires_grad = not frozen

        # Initialize the MLP for NNConv
        self.edge_mlp2 = EdgeMLP(num_edge_features, num_node_features, hidden_channels[0])
        
        # NNConv layer
        self.conv3 = NNConv(num_node_features, hidden_channels[0], self.edge_mlp2)
        
        # GATConv layer
        self.conv4 = GATConv(hidden_channels[0], hidden_channels[1], heads=1, concat=False)

        # First fully connected layer
        self.fc1 = nn.Linear(hidden_channels[1], hidden_channels[2])

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Last fully connected layer
        self.fc2 = nn.Linear(hidden_channels[2], 1)
        self.fc3 = nn.Linear(hidden_channels[2], 3)
        
        # Weight initialization
        self.apply(init_weights)
    
    def forward(self, batch, classify="binary", head="linear", dropout=True):

        # ECC1
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)

        # GAT
        x = self.conv2(x, batch.edge_index)
        x = F.relu(x)
        
        # ECC
        x = self.conv3(x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)

        # GAT
        x = self.conv4(x, batch.edge_index)
        x = F.relu(x)

        # Global average pooling
        x = global_mean_pool(x, batch.batch)


        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        if dropout:
            x = self.dropout(x)
        
        # Classification mode
        if classify == "binary":
            x = self.fc2(x)
            x = x.squeeze(1)
        if classify == "multiclass":
            x = self.fc3(x)

        # Prediction head
        if head == "linear":
            return x
        if head == "sigmoid":
            return torch.sigmoid(x)
        if head == "softmax":
            return torch.softmax(x, dim=1)


class downstream2(nn.Module):    
    def __init__(self, config, pretrained_layers, frozen=False):
        """
        Downstream model for seizure detection (binary or multiclass). Retrains the entire GNN embedder and does not add any additional layers.
        
        Args:
            config (dict): Dictionary containing the configuration of the model, containing hidden_channels which is a single value, and the dropout probability.
            pretrained_layers (tuple): Tuple containing the pretrained layers.
            frozen (bool): If True, the pretrained layers are frozen. If False, the pretrained layers are unfrozen.
        
        """    
        super(downstream2, self).__init__()
        hidden_channels = config["hidden_channels"]
        dropout = config["dropout"]
        
        # Pretrained layers
        EdgeMLP_pretrained, NNConv_pretrained, GATConv_pretrained = pretrained_layers
        self.conv1 = NNConv_pretrained
        self.conv2 = GATConv_pretrained

        # Assign the pretrained MLP to the NNConv1
        NNConv_pretrained.edge_mlp = EdgeMLP_pretrained
        
        # Output feature dimensions of pretrained layers
        num_node_features = GATConv_pretrained.out_channels
        num_edge_features = EdgeMLP_pretrained.state_dict()['mlp.0.weight'].size()[1]
        
        # Conditionally freeze or unfreeze pretrained layers
        for param in EdgeMLP_pretrained.parameters():
            param.requires_grad = not frozen
        for param in NNConv_pretrained.parameters():
            param.requires_grad = not frozen
        for param in GATConv_pretrained.parameters():
            param.requires_grad = not frozen

        # First fully connected layer
        self.fc1 = nn.Linear(num_node_features, hidden_channels)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Last fully connected layer
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.fc3 = nn.Linear(hidden_channels, 3)
        
        # Weight initialization
        self.apply(init_weights)
    
    def forward(self, batch, classify="binary", head="linear", dropout=True):
        
        # ECC1
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
        
        # Classification mode
        if classify=="binary":
            x = self.fc2(x)
            x = x.squeeze(1)
        if classify=="multiclass":
            x = self.fc3(x)

        # Prediction head
        if head=="linear":
            return x
        if head=="sigmoid":
            return torch.sigmoid(x)
        if head=="softmax":
            return torch.softmax(x, dim=1)


        
class CPC(nn.Module):
    def __init__(self):
        super(CPC, self).__init__()
    

class VICReg(nn.Module):
    def __init__(self):
        super(VICReg, self).__init__()