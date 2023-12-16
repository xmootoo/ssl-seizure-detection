import time
import json
import os
import torch
import torch.nn.functional as F
import random
import wandb
from loss import VICRegT1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from preprocess import run_sorter, combiner, create_data_loaders, extract_layers
from models import relative_positioning, temporal_shuffling, supervised_model, downstream1, downstream2, VICRegT1

os.environ["WANDB_INIT_TIMEOUT"] = "300"

def load_data(data_path, run_type="all", data_size=1.0):
    """
    Loads data from a folder of multiple runs.

    Args:
        data_path (str): Path to the data folder.
        run_type (str): Specifies which runs to load. Options are "all", "combined", or "runx" where x is the run number.
        data_size (float or int): The proportion of the full dataset to use.
    
    Returns:
        data (list): A list of graph representations for a certain number of runs, where the runs included depend on run_type.
    """
    
    data = run_sorter(data_path, run_type)
    
    # Compute total samples
    n = 0
    for run in data:
        n += len(run)

    if data_size <= 1.0:
        desired_samples = int(n * data_size)
    elif data_size > 1.0:
        desired_samples = data_size
    
    if run_type == "all":
        if data:  # Check if data is not empty
            data = combiner(data, desired_samples)
        else:
            print("Error. No data provided from run_sorter().")
            return None
    else:
        # Scale down data size and return
        random.shuffle(data)
        data = data[:desired_samples]

    return data


def forward_pass(model, batch, model_id="supervised", classify="binary", head="linear", dropout=0.1):
    if model_id=="supervised" or model_id=="downstream1" or model_id=="downstream2":
        return model(batch, classify, head, dropout)
    elif model_id=="relative_positioning" or model_id=="temporal_shuffling":
        return model(batch, head)
    elif model_id=="VICRegT1":
        return model(batch)


def train_model(model, train_loader, optimizer, criterion, device, classify="binary", head="linear", dropout=True, 
                model_id="supervised", timing=True):
    
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
        outputs = forward_pass(model, batch, model_id, classify, head, dropout)

        # Do not reshape output if batch size is 1

        if model_id != "VICRegT1":
            if outputs.shape[0] > 1:
                outputs = outputs.squeeze()
        
        # Select the correct label column based on classification type
        supervised_models = {"supervised", "downstream1", "downstream2"}
        if classify=="binary" and model_id in supervised_models:
            labels = batch.y[:, 0].float()  # Select the first column for binary classification
        elif classify=="multiclass" and model_id in supervised_models:
            labels = batch.y[:, 1].long().squeeze()  # Select the second column and ensure it's 1D
        else:
            labels = batch.y.float()
        
        # Calculate loss
        if model_id=="VICRegT1":
            loss = criterion(outputs[0].to(device), outputs[1].to(device), labels.to(device))
        else:
            loss = criterion(outputs.to(device), labels.to(device))
        epoch_train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Compute predictions for binary or multiclass classification
        if classify == "binary":
            probabilities = torch.sigmoid(outputs).squeeze()
            predictions = (probabilities > 0.5).float().to(device)
        elif classify == "multiclass":
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # Sum total correct predictions
        if model_id!="VICRegT1":
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)
        
        # Compute 100 batch timing seconds
        if timing:
            if batch_idx % 100 == 0:
                end_time = time.time()
                print("Time elapsed after 100 batches (training):", end_time - start_time)
                start_time = time.time()

    # Compute average loss and accuracy
    avg_loss = epoch_train_loss / len(train_loader)
    
    if model_id!="VICRegT1":
        accuracy = 100.0 * correct_train / total_train
        return avg_loss, accuracy
    else:
        return avg_loss, None


def evaluate_model(model, loader, criterion, device, classify="binary", head="linear", dropout=False, 
                   model_id="supervised", timing=True):
    
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
            outputs = forward_pass(model, batch, model_id, classify, head, dropout)
            
            # Do not reshape output if batch size is 1
            if model_id != "VICRegT1":
                if outputs.shape[0] > 1:
                    outputs = outputs.squeeze()

            # Select the correct label column based on classification type
            supervised_models = {"supervised", "downstream1", "downstream2"}
            if classify == "binary" and model_id in supervised_models:
                labels = batch.y[:, 0].float()  # Select the first column for binary classification
            elif classify == "multiclass" and model_id in supervised_models:
                labels = batch.y[:, 1].long().squeeze()  # Select the second column and ensure it's 1D
            else:
                labels = batch.y.float()
 
            # Calculate loss
            if model_id=="VICRegT1":
                loss = criterion(outputs[0].to(device), outputs[1].to(device), labels.to(device))
            else:
                loss = criterion(outputs.to(device), labels.to(device))
            epoch_eval_loss += loss.item()
            
            # Compute predictions for binary or multiclass classification
            if classify == "binary":
                probabilities = torch.sigmoid(outputs).squeeze()
                predictions = (probabilities > 0.5).float()
            elif classify == "multiclass":
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)


            # Sum total correct predictions
            if model_id!="VICRegT1":
                correct_eval += (predictions == labels).sum().item()
                total_eval += labels.size(0)
            
            if timing:
                if batch_idx % 100 == 0:
                    end_time = time.time()
                    print("Time elapsed after 100 batches (evaluation):", end_time - start_time)
                    start_time = time.time()

    avg_loss = epoch_eval_loss / len(loader)
    
    if model_id!="VICRegT1":
        accuracy = 100.0 * correct_eval / total_eval
        return avg_loss, accuracy
    else:
        return avg_loss, None


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


def train(data_path, logdir, patient_id, epochs, config, data_size=1.0, val_ratio=0.2, test_ratio=0.1, 
          batch_size=32, num_workers=4, lr=1e-3, weight_decay=1e-3, model_id="supervised", timing=True, 
          classify="binary", head="linear", dropout=True, datetime_id=None, run_type="all", frozen=False,
          model_path=None, model_dict_path=None, transfer_id=None, train_ratio=None, loss_config=None,
          project_id="Test Bay", patience = 20):
    """
    Trains the supervised GNN model, relative positioning model, or temporal shuffling model.

    Args:
        data_path (str): Path to the data file.
        logdir (str): Path to folder to dump model and stats data. 
        patient_id (str): Patient ID.
        epochs (int): Number of training iterations. 
        config (dict): Dictionary of model hyperparameters.
        data_size (float, optional): Proportion of total dataset to use. Defaults to 1.0.
        val_ratio (float, optional): If between 0 and 1, it is the proportion of data_size * len(data) to be used for validation. Defaults to 0.2.
                                     If > 1, then it is the exact number of examples to use for validation.
        test_ratio (float, optional): If between 0 and 1, it is the proportion of data_size * len(data) to be used for testing. Defaults to 0.1.
                                        If > 1, then it is the exact number of examples to use for testing.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for CPU. Defaults to 4.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-3.
        model_id (str, optional): Model identification corresponding to which model we are training. Options (from models.py): "supervised", "relative_positioning",
                                  "temporal_shuffling", "downstream1", and "downstream2". Defaults to "supervised".
        timing (bool, optional): Whether to time the training and evaluation per 100 batches. Defaults to True.
        classify (str, optional): Whether to use binary or multiclass classification. Defaults to "binary".
        head (str, optional): Whether to use a "linear", "sigmoid", or "softmax" head. Defaults to "linear". We typically use the linear head for numerical stability purposes
                              in tandem with the BCEWithLogitsLoss() loss function for binary classification, or the CrossEntropyLoss() loss function for multiclass classification.
        dropout (bool, optional): Whether to use dropout, fixed to p=0.1. Defaults to True.
        datetime_id (str, optional): The date and time of the training run. Defaults to None.
        run_type (str, optional): Specifies which runs to load. Options are "all", "combined", or "runx" where x is the run number. Defaults to "all".
                                  If "all" is selected, then all runs will be loaded and combined. The "combined" selection is only available for the list of
                                   PyG Data objects used for supervised learning, it is not available for pseudo-datasets.
        frozen (bool, optional): Whether to freeze the weights of the pretrained model. Defaults to False.
        model_path (str, optional): Path to the pretrained model. Defaults to None.
        model_dict_path (str, optional): Path to the pretrained model's state_dict. Defaults to None.
        transfer_id (str, optional): The model ID of the pretrained model, such as "relative_positioning" or "temporal_shuffling". Defaults to None.
        train_ratio (float, optional): Proportion of data_size * len(data) to be used for training if between 0 and 1, if > 1 then it is the exact 
                                        number of examples to use for training. Defaults to None. If set to None, then the remaining data not used in 
                                        validation and testing is used for training.
        loss_config (dict, optional): Dictionary of loss hyperparameters. Defaults to None.
        project_id (str, optional): Project ID for Weights & Biases. Defaults to None. Suggested: "ssl-seizure-detection-V2".
    
    Saves:
        model (pytorch model): PyTorch model architecture and weights.
        model_state_dict (pytorch model): PyTorch model parameters.
        train_loss (list): Training loss per epoch.
        val_loss (list): Validation loss per epoch.
        train_acc (list): Training accuracy per epoch.
        val_acc (list): Validation accuracy per epoch.
        test_loss (list): Test loss per epoch.
        test_acc (list): Test accuracy per epoch.
        info (txt): Training information.
    """
    
    # Initialize Weights & Biases
    wandb.init(project= project_id, config=config, name=f"{patient_id}_{model_id}_{datetime_id}_{run_type}")
    if transfer_id is not None:
        wandb.run.name = f"{patient_id}_{model_id}_{datetime_id}_{run_type}_{transfer_id}"
    
    # Load data
    data = load_data(data_path, run_type, data_size)
    
    # Assign GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Using MPS Device Acceleration.")
    else:
         print(f"Using device: {device}")
    
    # Initialize loaders, scaler, model, optimizer, and loss
    loaders, loader_stats = create_data_loaders(data, val_ratio=val_ratio, test_ratio=test_ratio, batch_size=batch_size, 
                                                num_workers=num_workers, model_id=model_id, train_ratio=train_ratio)
    if test_ratio != 0:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
    
    
    # Select model
    if model_id=="supervised":
        model = supervised_model(config).to(device)
    elif model_id=="relative_positioning":
        model = relative_positioning(config).to(device)
    elif model_id=="temporal_shuffling":
        model = temporal_shuffling(config).to(device)
    elif model_id=="downstream1":
        extracted_layers = extract_layers(model_path, model_dict_path, transfer_id) 
        model = downstream1(config, extracted_layers, frozen).to(device)
    elif model_id=="downstream2":
        extracted_layers = extract_layers(model_path, model_dict_path, transfer_id) 
        model = downstream2(config, extracted_layers, frozen).to(device)
    elif model_id=="VICRegT1":
        model = VICRegT1(config).to(device)
        
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.002)
    
    # Initialize loss based on classification method and head
    if classify=="binary" and head=="linear":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif classify=="binary" and head=="sigmoid":
        criterion = torch.nn.BCELoss()
    elif classify=="multiclass" and head=="linear":
        criterion = torch.nn.CrossEntropyLoss()
    elif classify=="multiclass" and head=="softmax":
        criterion = torch.nn.NLLLoss()
    if model_id=="VICRegT1":
        criterion = VICRegT1Loss(loss_config)
    
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
        epoch_train_loss, epoch_train_acc = train_model(model, train_loader, optimizer, criterion, device, classify, head, dropout, 
                                                        model_id, timing)
        
        #<----------Validation---------->
        epoch_val_loss, epoch_val_acc = evaluate_model(model, val_loader, criterion, device, classify, head, dropout=False, model_id=model_id, timing=timing)

        print(f'Epoch: {epoch+1}, Train Loss: {epoch_train_loss}, Train Accuracy: {epoch_train_acc}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_acc}')


        #<----------Statistics---------->
        train_loss.append((epoch, epoch_train_loss))
        val_loss.append((epoch, epoch_val_loss))
        train_acc.append((epoch, epoch_train_acc))
        val_acc.append((epoch, epoch_val_acc))

        # Weights & Biases Logging
        wandb.log({"Epoch": epoch+1, "Training Loss": epoch_train_loss, "Validation Loss": epoch_val_loss, 
                   "Training Accuracy": epoch_train_acc, "Validation Accuracy": epoch_val_acc})
        
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
        
        scheduler.step()
    
           
    #<----------Testing---------->
    if test_ratio!=0:
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, classify, head, dropout=False, model_id=model_id, 
                                             timing=timing)
        save_to_json(test_loss, stats_dir, "test_loss.json")
        save_to_json(test_acc, stats_dir, "test_acc.json")
        print(f"Training complete. Test Loss: {test_loss}. Test Accuracy: {test_acc}.")
        wandb.log({"Test Loss": test_loss, "Test Accuracy": test_acc})

    
    #<----------Save Statistics & Training Information---------->
    save_to_json(train_loss, stats_dir, "train_loss.json")
    save_to_json(val_loss, stats_dir, "val_loss.json")
    save_to_json(train_acc, stats_dir, "train_acc.json")
    save_to_json(val_acc, stats_dir, "val_acc.json")
    
    info_dict = {
        'Patient ID': patient_id,
        'Model ID': model_id,
        'Transfer ID': transfer_id,
        'Classify': classify,
        'Frozen': frozen,
        'Predictive Head': head,
        'Date & Time': datetime_id,
        'Data size': data_size,
        'Total examples': loader_stats["total_examples"],
        'Used examples': loader_stats["used_examples"],
        'Training examples': loader_stats["train_examples"], 
        'Validation examples': loader_stats["val_examples"],
        'Training batches': loader_stats["train_batches"],
        'Validation batches': loader_stats["val_batches"],
        'Validation ratio': val_ratio,
        'Test ratio': test_ratio,
        'Train ratio': train_ratio,
        'Batch size': batch_size,
        'Number of workers': num_workers,
        'Learning rate': lr,
        'Weight decay': weight_decay,
        'Number of epochs': epochs,
        'Model parameters': config,
        'Dropout': dropout,
    }
    
    if test_ratio!=0:
        info_dict['Test examples'] = loader_stats["test_examples"]
        info_dict['Test batches'] = loader_stats["test_batches"]
    
    # Weights & Biases Saving
    for key, value in info_dict.items():
        wandb.config[key] = value

    info = '\n'.join([f"{key}: {value}" for key, value in info_dict.items()])
    info_path = os.path.join(stats_dir, "info.txt")
    
    with open(info_path, "w") as f:
        f.write(info)
    
    print("Training complete.")
    
    # Weights & Biases finish the experiment
    wandb.finish()