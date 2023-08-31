import torch
import time
import json
from pyg_preprocess import create_data_loaders
from pyg_model import relative_positioning
from torch.cuda.amp import autocast, GradScaler


# def train( epochs, patience, load_path, save_path)

# Load data
path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_pseudolabeled\relative_positioning\PyG\jh101_12s_7min_PairData.pt"
data = torch.load(path)

# Assign GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_node_features = 1
num_edge_features = 1
hidden_channels = 64
out_channels = 32
epochs = 1
patience = 20  # Number of epochs to wait before early stop

if __name__ == '__main__':

    # Dataloaders
    train_loader, val_loader = create_data_loaders(data, data_size = 0.05)

    # Initialization
    scaler = GradScaler()
    model = relative_positioning(num_node_features, num_edge_features, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    counter = 0  # Counter for early stopping

    # Initialize lists to store loss and accuracy data as tuples
    train_loss_data = []
    val_loss_data = []
    train_acc_data = []
    val_acc_data = []

    for epoch in range(epochs):  # 100 is just an example, set your own number of epochs
        model.train()
        start_time = time.time()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        #<---------------------------------------Training--------------------------------------->
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch.to(device)
            with autocast():
                out = model(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch,
                            batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
                loss = criterion(out.squeeze(), batch.y.float().to(device))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()
            # Calculate training accuracy
            pred = torch.sigmoid(out)
            pred = (pred > 0.5).float()
            correct_train += (pred.squeeze() == batch.y.float().to(device)).sum().item()
            total_train += len(batch.y)
            print(batch_idx)
        epoch_train_loss /= len(train_loader)
        train_accuracy = 100. * correct_train / total_train
        train_loss_data.append((epoch, epoch_train_loss))
        train_acc_data.append((epoch, train_accuracy))

        #<---------------------------------------Validation--------------------------------------->
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch.to(device)
                with autocast():
                    out = model(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch,
                                batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
                    loss = criterion(out.squeeze(), batch.y.float().to(device))
                epoch_val_loss += loss.item()
                
                # Calculate validation accuracy
                pred = torch.sigmoid(out)
                pred = (pred > 0.5).float()
                correct_val += (pred.squeeze() == batch.y.float().to(device)).sum().item()
                total_val += len(batch.y)
                print(batch_idx)
                
        epoch_val_loss /= len(val_loader)
        val_accuracy = 100. * correct_val / total_val
        val_loss_data.append((epoch, epoch_val_loss))
        val_acc_data.append((epoch, val_accuracy))

        # Print epoch results
        print(f'Epoch: {epoch}, Train Loss: {epoch_train_loss}, Train Accuracy: {train_accuracy}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {val_accuracy}')

        # <--------------------------------------Early Stopping-------------------------------------->
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    #<------------------------Save stats to JSON------------------------>
    with open('train_loss_data.json', 'w') as f:
        json.dump(train_loss_data, f)
    with open('val_loss_data.json', 'w') as f:
        json.dump(val_loss_data, f)
    with open('train_acc_data.json', 'w') as f:
        json.dump(train_acc_data, f)
    with open('val_acc_data.json', 'w') as f:
        json.dump(val_acc_data, f)

