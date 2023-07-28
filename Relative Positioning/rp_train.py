import numpy as np
import torch
import torch.nn as nn
from torch import optim
from rp_model import GNN_encoder, Contrast, LogisticRegression
from rp_preprocess import dataloader


def train(model_enc, model_cont, model_logreg, data_loader, lr=0.001, num_epochs=10):
    optimizer = optim.Adam(
        list(model_enc.parameters()) + list(model_cont.parameters()) + list(model_logreg.parameters()),
        lr=lr
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in data_loader:
            X_1, X_2 = inputs[:, 0], inputs[:, 1]
            for i in range(len(X_1)):
                optimizer.zero_grad()

                # Forward pass through Net
                z_1, z_2 = model_enc(X_1[i]), model_enc(X_2[i])

                # Forward pass through Cont
                z_3 = model_cont(z_1, z_2)

                # Forward pass through LogisticRegression
                output = model_logreg(z_3)

                # Compute the loss
                loss = criterion(output, labels[i])
                total_loss += loss.item()

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")


# Hyperparameters for GNN
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

# Modules
model_enc = GNN_encoder(num_nodes, nf_dim, ef_dim, num_heads, GAT_dim, hidden_dim, final_dim)
model_cont = Contrast()
odel_logreg = LogisticRegression()


# Dummy iEEG graph representations




x = torch.randn(num_nodes, num_node_features)
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
edge_attr = torch.randn(num_edges, num_edge_features)

# Hyperparameters for dataloader
T = 30
tau_pos = 60
tau_neg = 120
batch_size = 16
step_size = 10
lr = 0.0022
num_epochs = 50

# Dataloader
data_loader = dataloader(A, T, step_size, tau_pos, tau_neg, batch_size)

# Dummy node features
NF = np.random.rand(22, 9)

# Dummy edge features
EF = np.random.rand(22, 22, 9)

# Loss
criterion = nn.BCELoss()

print("ok")

