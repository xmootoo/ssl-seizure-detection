import time
import json
import os
import torch
import torch.nn.functional as F
import random
import wandb
from ssl_seizure_detection.src.modules.loss import VICRegT1Loss
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ssl_seizure_detection.src.data.preprocess import run_sorter, combiner, create_data_loaders, extract_layers
from ssl_seizure_detection.src.modules.models import relative_positioning, temporal_shuffling, supervised, VICRegT1, downstream1, downstream2, downstream3


def load_data(config):
    """
    Loads data from a folder of multiple runs.

    Args:
        data_path (str): Path to the data folder.
        run_type (str): Specifies which runs to load. Options are "all", "combined", or "runx" where x is the run number.
        data_size (float or int): The proportion of the full dataset to use.
    
    Returns:
        data (list): A list of graph representations for a certain number of runs, where the runs included depend on run_type.
    """
    
    data = run_sorter(config.data_path, config.run_type)
    
    # Compute total samples
    n = 0
    for run in data:
        n += len(run)

    if config.data_size <= 1.0:
        desired_samples = int(n * config.data_size)
    elif config.data_size > 1.0:
        desired_samples = config.data_size
    
    if config.run_type == "all":
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


def forward_pass(model, batch, model_id="supervised", classify="binary", head="linear", p=0.1):
    if model_id=="supervised":
        return model(batch)
    elif model_id=="downstream1" or model_id=="downstream2":
        return model(batch, classify, head, p)
    elif model_id=="relative_positioning" or model_id=="temporal_shuffling":
        return model(batch, head)
    elif model_id=="VICRegT1" or model_id=="downstream3":
        return model(batch)


def get_labels(batch, classify=None, model_id=None):
    supervised_models = {"supervised", "downstream1", "downstream2", "downstream3"}
    if classify=="binary" and model_id in supervised_models:
        labels = batch.y[:, 0].float()  # Select the first column for binary classification
    elif classify=="multiclass" and model_id in supervised_models:
        labels = batch.y[:, 1].long()  # Select the second column and ensure it's 1D
    else:
        labels = batch.y.float()
    
    return labels


def get_loss(model_id, outputs, labels, criterion, device):
    if model_id=="VICRegT1":
        loss = criterion(outputs[0].to(device), outputs[1].to(device), labels.to(device))
    else:
        loss = criterion(outputs.to(device), labels.to(device))
    
    return loss

def get_predictions(classify, outputs, device):
    if classify == "binary":
        probabilities = torch.sigmoid(outputs).squeeze()
        predictions = (probabilities > 0.5).float().to(device)
    elif classify == "multiclass":
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    return predictions


def update_time(start_time, mode="training"):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed after 100 batches ({mode}): {elapsed_time:.2f}s")
    return end_time


def calculate_metrics(epoch_train_loss, correct_train, total_train, train_loader, model_id):

    avg_loss = epoch_train_loss / len(train_loader)
    
    if model_id != "VICRegT1":
        accuracy = 100.0 * correct_train / total_train if total_train > 0 else 0
        return avg_loss, accuracy
    else:
        return avg_loss, None
    

def process_model(config, model, loader, criterion, device, mode="training", optimizer=None):
    """
    Function to process the model, either in training or evaluation mode.

    Args:
        model (nn.Module): The model to be processed.
        loader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        classify (str): Classification type, 'binary' or 'multiclass'.
        head (str): Type of head used in model, 'linear', 'sigmoid', or 'softmax'.
        dropout (bool or float): Whether to use dropout or dropout rate.
        model_id (str): Identifier for the model.
        timing (bool): Whether to time the process.
        mode (str): 'training' or 'evaluation'.
        optimizer (torch.optim.Optimizer): Optimizer for training. Required if mode is 'training'.

    Returns:
        tuple: Tuple containing average loss and accuracy.
    """
    if mode == "training":
        model.train()
        if optimizer is None:
            raise ValueError("Optimizer is required for training mode")
    elif mode == "evaluation":
        model.eval()
    
    epoch_loss, correct, total = 0, 0, 0
    if config.timing:
        start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.set_grad_enabled(mode == "training"):
            outputs = forward_pass(model, batch, config.model_id, config.classify, config.head)
            labels = get_labels(batch, config.classify, config.model_id)
            loss = get_loss(config.model_id, outputs, labels, criterion, device)
            epoch_loss += loss.item()

            if mode == "training":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if config.classify:
                predictions = get_predictions(config.classify, outputs, device)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if config.timing and batch_idx % 100 == 0:
            start_time = update_time(start_time, mode=mode)
    
    avg_loss, accuracy = calculate_metrics(epoch_loss, correct, total, loader, config.model_id)
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



def initialize_model(config, model_config, device):
    """ 
    Initializes the model based on the model ID.

    """
    
    if config.model_id=="supervised":
        model = supervised(model_config).to(device)
    elif config.model_id=="relative_positioning":
        model = relative_positioning(model_config).to(device)
    elif config.model_id=="temporal_shuffling":
        model = temporal_shuffling(model_config).to(device)
    elif config.model_id.startswith("downstream"):
        model_class = eval(config.model_id) 
        extracted_layers = extract_layers(config.model_path, config.model_dict_path, config.transfer_id) 
        model = model_class(config=model_config, pretrained_layers=extracted_layers, requires_grad=config.requires_grad).to(device)
    elif config.model_id=="VICRegT1":
        model = VICRegT1(model_config).to(device)
    
    return model


def initialize_optimizer(model, config):
    """
    Initialize the optimizer with different learning rates for the encoder and classifier
    parts of the downstream model, if the model_id starts with 'downstream'.

    Args:
        model (downstream3): The downstream3 model instance.
        model_id (str): Identifier for the model.
        lr (list): A list where the first element is the learning rate for the encoder and the 
                   second element is the learning rate for the classifier.
        weight_decay (float): Weight decay factor for regularization.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """

    # For downstream and supervised comparison
    if type(config.lr) == list:
        lr_encoder, lr_classifier = config.lr  # Unpack the learning rates

        # Create parameter groups for encoder and classifier
        encoder_params = {'params': model.encoder.parameters(), 'lr': lr_encoder}
        classifier_params = {'params': model.classifier.parameters(), 'lr': lr_classifier}

        # Initialize the optimizer with these parameter groups
        optimizer = optim.Adam([encoder_params, classifier_params], weight_decay=config.weight_decay)
    
    # Stanrd optimizer for SSL models
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return optimizer


def initialize_criterion(config, loss_config):
    
    if config.classify=="binary" and config.head=="linear":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config.classify=="binary" and config.head=="sigmoid":
        criterion = torch.nn.BCELoss()
    elif config.classify=="multiclass" and config.head=="linear":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.classify=="multiclass" and config.head=="softmax":
        criterion = torch.nn.NLLLoss()
    if config.model_id=="VICRegT1":
        criterion = VICRegT1Loss(loss_config)
    
    if criterion is None:
        raise ValueError("Invalid configuration for 'classify', 'head', or 'model_id'")

    return criterion


def initialize_wandb(config):
    
    # Initialize Weights & Biases
    wandb.init(project=config.project_id, config=config, name=f"{config.patient_id}_{config.model_id}_{config.datetime_id}_{config.run_type}")
    if config.transfer_id is not None:
        wandb.run.name = f"{config.patient_id}_{config.model_id}_{config.datetime_id}_{config.run_type}_{config.transfer_id}"


def initialize_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Using MPS Device Acceleration.")
    else:
        print(f"Using device: {device}")
    return device

def initialize_loaders(data, config):
    loaders, loader_stats = create_data_loaders(data, config)
    if config.test_ratio != 0:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = None
    
    return train_loader, val_loader, test_loader, loader_stats


def create_model_stats_dir(config):
    
    model_dir = os.path.join(config.logdir, 'model')
    stats_dir = os.path.join(config.logdir, 'stats')
    
    # Ensure directories exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    return model_dir, stats_dir

def wandb_log(epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc):
    wandb.log({"Epoch": epoch+1, "Training Loss": epoch_train_loss, "Validation Loss": epoch_val_loss, 
                   "Training Accuracy": epoch_train_acc, "Validation Accuracy": epoch_val_acc})
    return

def early_stopping(epoch_val_loss, best_val_loss, counter, model, model_dir, config):
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        counter = 0
        save_model(model, model_dir, config.model_id)
        return False
    else:
        counter += 1
        if counter >= config.patience:
            print("Early stopping triggered.")
            return True

def print_learning_rate(config, optimizer):
    if config.model_id in {"supervised", "downstream3"}:
            encoder_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            print(f"New encoder lr: {encoder_lr:.6f}. New classifier lr: {classifier_lr:.6f}")
    else:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"New lr: {current_lr:.6f}")

def testing_and_logging(config, model, test_loader, criterion, device, optimizer, stats_dir):
    test_loss, test_acc = process_model(config, model, test_loader, criterion, device, "evaluation", optimizer)
        
    save_to_json(test_loss, stats_dir, "test_loss.json")
    save_to_json(test_acc, stats_dir, "test_acc.json")
    print(f"Training complete. Test Loss: {test_loss}. Test Accuracy: {test_acc}.")
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_acc})

def save_stats(train_loss, val_loss, train_acc, val_acc, stats_dir):
    save_to_json(train_loss, stats_dir, "train_loss.json")
    save_to_json(val_loss, stats_dir, "val_loss.json")
    save_to_json(train_acc, stats_dir, "train_acc.json")
    save_to_json(val_acc, stats_dir, "val_acc.json")

def get_wandb_info(config, model_config, loss_config, loader_stats):
        info_dict = {
            'Patient ID': config.patient_id,
            'Model ID': config.model_id,
            'Transfer ID': config.transfer_id,
            'Experiment ID': config.exp_id,
            'Classify': config.classify,
            'Frozen (Encoder)': not config.requires_grad,
            'Predictive Head': config.head,
            'Date & Time': config.datetime_id,
            'Data size': config.data_size,
            'Total examples': loader_stats["total_examples"],
            'Used examples': loader_stats["used_examples"],
            'Training examples': loader_stats["train_examples"], 
            'Validation examples': loader_stats["val_examples"],
            'Training batches': loader_stats["train_batches"],
            'Validation batches': loader_stats["val_batches"],
            'Validation ratio': config.val_ratio,
            'Test ratio': config.test_ratio,
            'Train ratio': config.train_ratio,
            'Batch size': config.batch_size,
            'Number of workers': config.num_workers,
            'Learning rate': config.lr,
            'Weight decay': config.weight_decay,
            'Number of epochs': config.epochs,
            'Model parameters': config,
            'Dropout': model_config.dropout,
            'Dropout Probability': model_config.p,
            'Loss Config': loss_config,
        }
        return info_dict

def save_final_json(info_dict, stats_dir):
    info = '\n'.join([f"{key}: {value}" for key, value in info_dict.items()])
    info_path = os.path.join(stats_dir, "info.txt")
    
    with open(info_path, "w") as f:
        f.write(info)
