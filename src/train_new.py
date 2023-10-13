import torch
import time
import json
import os
import inspect
from preprocess import create_data_loaders
from models import relative_positioning, temporal_shuffling, supervised_model
from torch.cuda.amp import autocast, GradScaler

def load_data(path):
    return torch.load(path)

def forward_pass(model, batch, model_id="supervised", classify="linear", dropout=0.1):
    if model_id=="supervised":
        return model(batch, classify, dropout=dropout)
    elif model_id=="relative_positioning" or model_id=="temporal_shuffling":
        return model(batch, classify)

def train_model(model, train_loader, optimizer, criterion, device, model_id="supervised", timing=True):
    
    model.train()
    
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0
    
    if timing:
        start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Send batch to device
        batch = batch.to(device)

        # Compute forward pass
        outputs = forward_pass(model, batch, model_id)
        
        # Calculate loss
        loss = criterion(outputs.squeeze().to(device), batch.y.float().to(device))

        # Training statistics
        epoch_train_loss += loss.item()
        
        predictions = torch.sigmoid(outputs).squeeze().to(device)
        predictions = (predictions > 0.5).float()

        correct_train += (predictions == batch.y.float().to(device)).sum().item()
        total_train += len(batch.y)
        
        if timing:
            if batch_idx % 100 == 0:
                end_time = time.time()
                print("Time elapsed after 100 batches (training):", end_time - start_time)
                start_time = time.time()

    avg_loss = epoch_train_loss / len(train_loader)
    accuracy = 100.0 * correct_train / total_train

    return avg_loss, accuracy

def evaluate_model(model, loader, criterion, device, model_id="supervised", timing=True):
    
    model.eval()
    
    epoch_eval_loss = 0
    correct_eval = 0
    total_eval = 0
    
    with torch.no_grad():
        if timing:
            start_time = time.time()
        
        for batch_idx, batch in enumerate(loader):
            
            # Send batch to device
            batch = batch.to(device)
            
            # Compute forward pass
            outputs = forward_pass(model, batch, model_id)
            
            # Calculate loss
            loss = criterion(outputs.squeeze().to(device), batch.y.float().to(device))
            
            # Evaluation statistics
            epoch_eval_loss += loss.item()
            
            predictions = torch.sigmoid(outputs).squeeze().to(device)
            predictions = (predictions > 0.5).float()

            correct_eval += (predictions == batch.y.float().to(device)).sum().item()
            total_eval += len(batch.y)
            
            if timing:
                if batch_idx % 100 == 0:
                    end_time = time.time()
                    print("Time elapsed after 100 batches (evaluation):", end_time - start_time)
                    start_time = time.time()

    avg_loss = epoch_eval_loss / len(loader)
    accuracy = 100.0 * correct_eval / total_eval

    return avg_loss, accuracy




def save_model(model, logdir, model_name):
    """Saves PyTorch model architecture and weights, and model parameters (state_dict).

    Args:
        model (pytorch model): PyTorch model to save.
        path (string): Path to save the model.
        model_name (string): The name of the model. Include patient ID, hyperparameters for pseudodataset, number of epochs, etc.
    """
    
    # Save entire model architecture and weights
    model_dir = os.path.join(logdir, model_name + ".pth")
    torch.save(model, model_dir)

    # Save only the model parameters (state_dict)
    modeldic_dir = os.path.join(logdir, model_name + "_state_dict.pth")
    torch.save(model.state_dict(), modeldic_dir)
    

def save_to_json(data, logdir, file_name):
    """
    Save data to a JSON file.

    Args:
        data: Data to save.
        folder_path: The folder in which to save the file.
        file_name: The name of the file to save the data in.

    Returns:
        None
    """
    
    full_path = os.path.join(logdir, file_name)
    with open(full_path, 'w') as f:
        json.dump(data, f)



def train(data_path, logdir, patient_id, epochs, model_parameters, data_size=1.0, val_ratio=0.2, test_ratio=0.1, 
          batch_size=32, num_workers=4, lr=1e-3, model_id="supervised", timing=True):
    """Trains a the Relative Positioning SSL model.

    Args:
        data_path (str): Path to the data file.
        save_path (str): Path to folder to dump model and stats data. 
        epochs (int): Number of training iterations. 
        data_size (float, optional): Proportion of total dataset to use. Defaults to 1.0.
        train_ratio (float, optional): Proportion of data_size * len(dataset) to use for training, the complement is used for validation. Defaults to 0.8.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for CPU. Defaults to 4.
        lr (float, optional): Learning rate. Defaults to 1e-3.
    """
    
    # Load data
    data = load_data(data_path)
    
    # Assign GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Using MPS Device Acceleration.")
    else:
         print(f"Using device: {device}")


    # Patience parameter
    patience = 20

    # Initialize loaders, scaler, model, optimizer, and loss
    train_loader, val_loader, test_loader = create_data_loaders(data, data_size=data_size, val_ratio=val_ratio, test_ratio=test_ratio, 
                                                   batch_size=batch_size, num_workers=num_workers, model_id=model_id)


    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    print("Number of test batches:", len(test_loader))
    
    if model_id=="supervised":
        model = supervised_model(model_parameters).to(device)
    if model_id=="relative_positioning":
        model = relative_positioning(model_parameters).to(device)
    if model_id=="temporal_shuffling":
        model = temporal_shuffling(model_parameters).to(device)

    print(f"Model information: {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training statistics
    best_val_loss = float('inf')
    counter = 0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # Create directory to save model and stats
    model_dir = os.path.join(logdir, 'model')
    stats_dir = os.path.join(logdir, 'stats')
    
    # Ensure directories exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    # Train our model for multiple epochs
    for epoch in range(epochs):
        
        #<----------Training---------->
        epoch_train_loss, epoch_train_acc = train_model(model, train_loader, optimizer, criterion, device, model_id, timing)
        
        #<----------Validation---------->
        epoch_val_loss, epoch_val_acc = evaluate_model(model, val_loader, criterion, device, model_id, timing)

        print(f'Epoch: {epoch+1}, Train Loss: {epoch_train_loss}, Train Accuracy: {epoch_train_acc}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_acc}')


        #<----------Statistics---------->
        train_loss.append((epoch, epoch_train_loss))
        val_loss.append((epoch, epoch_val_loss))
        train_acc.append((epoch, epoch_train_acc))
        val_acc.append((epoch, epoch_val_acc))

        #<----------Early Stopping---------->
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            save_model(model, model_dir, model_id)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
        
    #<----------Testing---------->
    if model_id=="supervised":
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, model_id, timing=False)
        save_to_json(test_loss, stats_dir, "test_loss.json")
        save_to_json(test_acc, stats_dir, "test_acc.json")
        print(f"Epoch: {epoch+1}. Test Loss: {test_loss}. Test Accuracy: {test_acc}.")
            
    #<----------Save Statistics---------->
    save_to_json(train_loss, stats_dir, "train_loss.json")
    save_to_json(val_loss, stats_dir, "val_loss.json")
    save_to_json(train_acc, stats_dir, "train_acc.json")
    save_to_json(val_acc, stats_dir, "val_acc.json")
    
    info = f"Patient ID: {patient_id}\nData size: {data_size}\nValidation ratio: {val_ratio}\nTest ratio: {test_ratio}\nBatch size: {batch_size}\nNumber of workers: {num_workers}\nLearning rate: {lr}\nNumber of epochs: {epochs}\nModel parameters: {model_parameters}"
    info_path = os.path.join(stats_dir, "info.txt")

    with open(info_path, "w") as f:
        f.write(info)
    
    print("Training complete.")