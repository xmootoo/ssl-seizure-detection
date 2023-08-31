import torch
import time
import json
from pyg_preprocess import create_data_loaders
from pyg_model import relative_positioning
from torch.cuda.amp import autocast, GradScaler

def load_data(path):
    return torch.load(path)

def train_model(model, train_loader, optimizer, scaler, criterion, device):
    model.train()
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        with autocast():
            out = model(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch,
                        batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
            loss = criterion(out.squeeze(), batch.y.float().to(device))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_train_loss += loss.item()
        pred = torch.sigmoid(out)
        pred = (pred > 0.5).float()
        correct_train += (pred.squeeze() == batch.y.float().to(device)).sum().item()
        total_train += len(batch.y)
        
        if batch_idx % 100 == 0:
            end_time = time.time()
            print("Time Elapse after 100 batches (training):", end_time - start_time)
            start_time = time.time()
        
    return epoch_train_loss / len(train_loader), 100. * correct_train / total_train

def validate_model(model, val_loader, criterion, device):
    model.eval()
    epoch_val_loss = 0
    correct_val = 0
    total_val = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)
            with autocast():
                out = model(batch.x1, batch.edge_index1, batch.edge_attr1, batch.x1_batch,
                            batch.x2, batch.edge_index2, batch.edge_attr2, batch.x2_batch)
                loss = criterion(out.squeeze(), batch.y.float().to(device))
            epoch_val_loss += loss.item()
            pred = torch.sigmoid(out)
            pred = (pred > 0.5).float()
            correct_val += (pred.squeeze() == batch.y.float().to(device)).sum().item()
            total_val += len(batch.y)

            if batch_idx % 100 == 0:
                end_time = time.time()
                print("Time Elapse after 100 batches (validation):", end_time - start_time)
                start_time = time.time()
        
    return epoch_val_loss / len(val_loader), 100. * correct_val / total_val


def save_model(model, path, model_name):
    """Saves PyTorch model architecture and weights, and model parameters (state_dict).

    Args:
        model (pytorch model): PyTorch model to save.
        path (string): Path to save the model.
        model_name (string): The name of the model. Include patient ID, hyperparameters for pseudodataset, number of epochs, etc.
    """
    
    # Save entire model architecture and weights
    torch.save(model, path + model_name +".pth")

    # Save only the model parameters (state_dict)
    torch.save(model.state_dict(), path + model_name + "_state_dict.pth")
    


if __name__ == '__main__':
    
    # Timer
    start_time = time.time()
    
    # Load data
    path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_pseudolabeled\relative_positioning\PyG\jh101_12s_7min_PairData.pt"
    data = load_data(path)
    
    # Save directory and model name
    save_dir = r"C:\Users\xmoot\Desktop\Models\PyTorch\ssl-seizure-detection\relative_positioning\jh101"
    model_name = r"\jh101_12s_7min_1epochs"
    
    # Assign GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model and data parameters
    num_node_features = 1
    num_edge_features = 1
    hidden_channels = 64
    out_channels = 32
    epochs = 1
    patience = 20

    # Initialize loaders, scaler, model, optimizer, and loss
    train_loader, val_loader = create_data_loaders(data, data_size=1.0)
    print(len(train_loader), len(val_loader))
    scaler = GradScaler()
    model = relative_positioning(num_node_features, num_edge_features, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training statistics
    best_val_loss = float('inf')
    counter = 0
    train_loss_data = []
    val_loss_data = []
    train_acc_data = []
    val_acc_data = []

    # Train our model for multiple epochs
    for epoch in range(epochs):
        
        #<----------Training---------->
        epoch_train_loss, train_accuracy = train_model(model, train_loader, optimizer, scaler, criterion, device)
        
        #<----------Validation---------->
        epoch_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        print(f'Epoch: {epoch}, Train Loss: {epoch_train_loss}, Train Accuracy: {train_accuracy}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {val_accuracy}')

        #<----------Statistics---------->
        train_loss_data.append((epoch, epoch_train_loss))
        val_loss_data.append((epoch, epoch_val_loss))
        train_acc_data.append((epoch, train_accuracy))
        val_acc_data.append((epoch, val_accuracy))

        #<----------Early Stopping---------->
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            save_model(model, save_dir, model_name)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
            
    #<----------Save Statistics---------->
    with open('train_loss_data.json', 'w') as f:
        json.dump(train_loss_data, f)
    with open('val_loss_data.json', 'w') as f:
        json.dump(val_loss_data, f)
    with open('train_acc_data.json', 'w') as f:
        json.dump(train_acc_data, f)
    with open('val_acc_data.json', 'w') as f:
        json.dump(val_acc_data, f)
