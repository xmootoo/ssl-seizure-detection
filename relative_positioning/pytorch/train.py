import torch
import time
import json
import os
from preprocess import create_data_loaders
from models import relative_positioning
from torch.cuda.amp import autocast, GradScaler

def load_data(path):
    return torch.load(path)

def train_model(model, train_loader, optimizer, scaler, criterion, device, timing=True):
    
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
        
        if timing:
            if batch_idx % 100 == 0:
                end_time = time.time()
                print("Time elapsed after 100 batches (training):", end_time - start_time)
                start_time = time.time()
        
    return epoch_train_loss / len(train_loader), 100. * correct_train / total_train

def validate_model(model, val_loader, criterion, device, timing=True):
    
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

            if timing:
                if batch_idx % 100 == 0:
                    end_time = time.time()
                    print("Time elapsed after 100 batches (validation):", end_time - start_time)
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
    

def save_to_json(data, save_path, file_name):
    """
    Save data to a JSON file.

    Args:
        data: Data to save.
        folder_path: The folder in which to save the file.
        file_name: The name of the file to save the data in.

    Returns:
        None
    """
    
    full_path = os.path.join(save_path, file_name)
    with open(full_path, 'w') as f:
        json.dump(data, f)



def train(data_path, model_path, stats_path, model_name, epochs, data_size=1.0, train_ratio=0.8, batch_size=32, num_workers=4,
          lr=1e-3, timing=True):
    """Trains a the Relative Positioning SSL model.

    Args:
        data_path (str): Path to the data file.
        save_path (str): Path to folder to dump model and stats data.
        model_name (str): Name of the model. Include patient ID, hyperparameters for pseudodataset, number of epochs, etc.
        epochs (int): Number of training iterations. 
        data_size (float, optional): Proportion of total dataset to use. Defaults to 1.0.
        train_ratio (float, optional): Proportion of data_size * len(dataset) to use for training, the complement is used for validation. Defaults to 0.8.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for CPU. Defaults to 4.
        lr (float, optional): Learning rate. Defaults to 1e-3.
    """
    
    #Start timer
    start_time = time.time()
    
    # Load data
    data = load_data(data_path)
    
    # Assign GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model and data parameters
    num_node_features = 1
    num_edge_features = 1
    hidden_channels = 64
    out_channels = 32
    patience = 20

    # Initialize loaders, scaler, model, optimizer, and loss
    train_loader, val_loader = create_data_loaders(data, data_size, train_ratio, batch_size, num_workers)
    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    scaler = GradScaler()
    model = relative_positioning(num_node_features, num_edge_features, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
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
        epoch_train_loss, train_accuracy = train_model(model, train_loader, optimizer, scaler, criterion, device, timing)
        
        #<----------Validation---------->
        epoch_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device, timing)

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
            save_model(model, model_path, model_name)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
            
    #<----------Save Statistics---------->
    save_to_json(train_loss_data, stats_path, "train_loss_data.json")
    save_to_json(val_loss_data, stats_path, "val_loss_data.json")
    save_to_json(train_acc_data, stats_path, "train_acc_data.json")
    save_to_json(val_acc_data, stats_path, "val_acc_data.json")