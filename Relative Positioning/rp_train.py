import numpy as np
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




# Filtered iEEG data
A = np.random.rand(22, 400)

# Hyperparameters
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

# Model
# model_enc = GNN_embedder()
# model_cont = Contrast()
# model_logreg = LogisticRegression()
