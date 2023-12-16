# Model
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm as GraphBatchNorm
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

   

class gnn_embedder2(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, batch_norm=True, dropout=True, p=0.1):
        super(gnn_embedder2, self).__init__()
        """
        The embedding architecture used in RP, TS, and VICRegT models. Comprised of an encoder module and projector module.
        
        """
        
        # Initialize the MLP for NNConv
        self.edge_mlp = EdgeMLP(num_edge_features, num_node_features, hidden_channels[0])
        
        # Encoder
        self.conv1 = NNConv(num_node_features, hidden_channels[0], self.edge_mlp)
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1], heads=1, concat=False)
        self.conv3 = GATConv(hidden_channels[1], hidden_channels[2], heads=1, concat=False)
        
        # Dropout
        self.dropout = dropout
        if self.dropout:
            self.net_dropout = nn.Dropout(p=p)
        else:
            self.net_dropout = nn.Identity()
        
        # Projector
        self.fc1 = nn.Linear(hidden_channels[2], hidden_channels[3])
        self.fc2 = nn.Linear(hidden_channels[3], hidden_channels[4])
        self.fc3 = nn.Linear(hidden_channels[4], hidden_channels[5])
        
        
        if batch_norm:
            # Batch Normalization for graph layers
            self.bn_graph1 = GraphBatchNorm(hidden_channels[0])
            self.bn_graph2 = GraphBatchNorm(hidden_channels[1])
            self.bn_graph3 = GraphBatchNorm(hidden_channels[2])
            
            # Batch Normalization for fully connected layers
            self.bn1 = nn.BatchNorm1d(hidden_channels[3])
            self.bn2 = nn.BatchNorm1d(hidden_channels[4])
        
        else:
            self.bn_graph1 = self.bn_graph2 = self.bn_graph3 = self.bn1 = self.bn2 = nn.Identity()
            
        # Weight initialization
        self.apply(init_weights)
        
    def forward(self, x, edge_index, edge_attr, batch):
        
        # NNConv layer
        x = F.relu(self.bn_graph1(self.conv1(x, edge_index, edge_attr)))
        
        # GATConv layers
        x = F.relu(self.bn_graph2(self.conv2(x, edge_index)))
        x = F.relu(self.bn_graph3(self.conv3(x, edge_index)))
        
        # Global average pooling
        x = global_mean_pool(x, batch) #<-- batch vector to keep track of graphs

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.net_dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.net_dropout(x)
        
        x = self.fc3(x)
        
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
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))

        # GAT
        x = F.relu(self.conv2(x, batch.edge_index))

        # Global average pooling
        x = global_mean_pool(x, batch.batch)


        # Fully connected layers
        x = F.relu(self.fc1(x))

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


class VICRegT1(nn.Module):
    def __init__(self, config):
        super(VICRegT1, self).__init__()
        num_node_features = config["num_node_features"]
        num_edge_features = config["num_edge_features"]
        hidden_channels = config.get("hidden_channels", [64, 128, 128, 512, 512, 512])
        batch_norm = config.get("batch_norm", True)
        dropout = config.get("dropout", True)
        p = config.get("p", 0.1)

        # GNN embedders
        self.embedder = gnn_embedder2(num_node_features, num_edge_features, hidden_channels, batch_norm, dropout, p)
        
        # Weight initialization
        self.apply(init_weights)
        
    def forward(self, batch):
        # Graph embeddings
        z1 = self.embedder(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch)
        z2 = self.embedder(batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)

        return (z1, z2)



def set_requires_grad(layers_dict, requires_grad=True):
    """
    Set the requires_grad attribute for all parameters in the layers contained in the given dictionary.

    Args:
        layers_dict (dict): A dictionary where keys are layer names and values are nn.Modules whose parameters will be set.
        requires_grad (bool): Whether the layers' parameters should require gradients (unfrozen) or not (frozen).
    """
    for layer in layers_dict.values():
        for param in layer.parameters():
            param.requires_grad = requires_grad



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

        # ECC 1
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))

        # GAT 1
        x = F.relu(self.conv2(x, batch.edge_index))
        
        # ECC 2
        x = F.relu(self.conv3(x, batch.edge_index, batch.edge_attr))
        
        # GAT 2
        x = F.relu(self.conv4(x, batch.edge_index))

        # Global average pooling
        x = global_mean_pool(x, batch.batch)


        # Fully connected layer
        x = F.relu(self.fc1(x))

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
        
        # ECC
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))

        # GAT
        x = F.relu(self.conv2(x, batch.edge_index))

        # Global average pooling
        x = global_mean_pool(x, batch.batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))

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


class downstream3(nn.Module):    
    def __init__(self, config, pretrained_layers, requires_grad=False):
        """
        Downstream model for seizure detection (binary or multiclass). Trains a GNN encoder (frozen or unfrozen) and a simple nonlinear classifier ontop
        (logistic regression or multinomial logistic regression) with frozen encoder or unfrozen encoder.
        
        Args:
            config (dict): Dictionary containing the configuration of the model, containing hidden_channels which is a single value, and the dropout probability.
            pretrained_layers (tuple): Tuple containing the pretrained layers.
            requires_grad (bool): Whether to require gradients for pretrained layers. If True, the layers are unfrozen, if False the layers are frozen.
        
        """    
        super(downstream3, self).__init__()
        hidden_channels = config["hidden_channels"]
        dropout = config["dropout"]

        # Graph layers (petrained)
        self.conv1 = pretrained_layers["conv1"]
        self.conv2 = pretrained_layers["conv2"]
        self.conv3 = pretrained_layers["conv3"]

        # Batch normalization layers (petrained)
        self.bn_graph1 = pretrained_layers["bn_graph1"]
        self.bn_graph2 = pretrained_layers["bn_graph2"]
        self.bn_graph3 = pretrained_layers["bn_graph3"]
        
        # Batch Normalization for fully connected layers
        self.bn1 = pretrained_layers["bn1"]
        self.bn2 = pretrained_layers["bn2"]

        # Assign the pretrained MLP to the NNConv1
        self.conv1.edge_mlp = pretrained_layers["edge_mlp"]
        
        # Output feature dimensions of pretrained layers
        num_node_features = self.conv1.out_channels
        num_edge_features = self.conv1.edge_mlp.state_dict()['mlp.0.weight'].size()[1]
        
        # Conditionally freeze or unfreeze pretrained layers
        set_requires_grad(pretrained_layers, requires_grad)

        # Dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        # Last fully connected layer
        self.fc1 = nn.Linear(num_node_features, 1)
        self.fc2 = nn.Linear(num_node_features, 3)
        
        # Weight initialization
        self.apply(init_weights)
    
    def forward(self, batch, classify="binary", head="linear", dropout=True):
        
        # ECC
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))

        # GAT Layers
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.relu(self.conv3(x, batch.edge_index))

        # Global average pooling
        x = global_mean_pool(x, batch.batch)
        
        # Classification mode
        if classify=="binary":
            x = self.fc1(x)
            x = x.squeeze(1)
        if classify=="multiclass":
            x = self.fc2(x)
        
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
    

