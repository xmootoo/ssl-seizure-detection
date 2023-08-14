import numpy as np
import torch
import torch.nn as nn
from torch import optim
from rp_model import GNN_encoder, Contrast, Regression
from dataloader import dataloaders_torch

# Hyperparameters
num_nodes = 8
nf_dim = (4, 12)
ef_dim = 32
GAT_dim = 128
num_heads = 1
final_dim = 64

# Learning rate
lr = 0.001

# L2 regularization strength
l2_reg = 0.01

# Number of epochs
num_epochs = 100

# Batch size
batch_size = 32

# Initialize the models
model_enc = GNN_encoder(num_nodes, nf_dim, ef_dim, GAT_dim, num_heads, final_dim)
model_cont = Contrast()
model_logreg = Regression(final_dim)

# Loss BCE (with logits) for numerical stability
criterion = nn.BCEWithLogitsLoss()


# Optimizer with L2 regularization
optimizer = optim.Adam(
    list(model_enc.parameters()) + list(model_cont.parameters()) + list(model_logreg.parameters()),
    lr=0.001,
    weight_decay=l2_reg  # L2 regularization
)



def train(model_enc, model_cont, model_logreg, data_loader, lr=0.001, num_epochs=10):
    """
    Train a graph neural network version of the relative positioning model.

    Args:
        model_enc (GNN_encoder): The graph neural network encoder module.
        model_cont (Contrast): The contrastive module.
        model_logreg (Regression): The linear regression module.
        data_loader (DataLoader): The data loader that loads the graph pairs.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        num_epochs (int, optional): The number of training epochs. Defaults to 10.
    """
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (batch, labels) in enumerate(data_loader):
            optimizer.zero_grad()

            # Convert Batch back into list of Data objects
            data_list = batch.to_data_list()

            # Separate list into two lists for the two graphs in each pair
            graph1_list = data_list[::2]  # Elements at even indices
            graph2_list = data_list[1::2]  # Elements at odd indices

            # Process each pair of graphs
            for graph1, graph2 in zip(graph1_list, graph2_list):
                
                # Forward pass
                z_1, z_2 = model_enc(graph1), model_enc(graph2)
                x = model_cont(z_1, z_2)
                logits = model_logreg(x)
                
                # Loss
                loss = criterion(logits.view(-1), labels.float())
                total_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print the loss every 5 epochs
                if epoch % 5 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")


# # Test case
# num_graphs = 10

# # Generate random edge indices, node features, and edge features
# edge_index = []
# edge_index = []
# node_features = []
# edge_features = []
# for _ in range(num_graphs):
#     num_edges1, num_edges2 = np.random.randint(1, num_nodes, size=2)
#     edge_index.append((np.random.randint(num_nodes, size=(2, num_edges1)), np.random.randint(num_nodes, size=(2, num_edges2))))
#     node_features.append((np.random.rand(num_nodes, nf_dim[0]), np.random.rand(num_nodes, nf_dim[0])))
#     edge_features.append((np.random.rand(num_edges1, ef_dim), np.random.rand(num_edges2, ef_dim)))

# # Generate random labels (binary classification)
# labels = np.random.randint(2, size=num_graphs)

# # Call your data loading function
# data_loader = rp_dataloader(edge_index, node_features, edge_features, labels)

# # Training
# train(model_enc, model_cont, model_logreg, data_loader, lr=0.001, num_epochs=2)