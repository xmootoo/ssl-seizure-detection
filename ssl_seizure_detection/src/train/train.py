
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from ssl_seizure_detection.src.train.utils import *


os.environ["WANDB_INIT_TIMEOUT"] = "300"

def train(config, model_config, loss_config):
    """
    Trains the supervised GNN model, relative positioning model, or temporal shuffling model.

    Args:
        data_path (str): Path to the data folder containing runs for a single patient.
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
        requires_grad (bool, optional): Whether to freeze the weights of the pretrained encoder. False = frozen and True = Unfrozen. Defaults to True.
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
    initialize_wandb(config)
    
    # Load data
    data = load_data(config)
    
    # Initialize device
    device = initialize_device()
    
    # Initialize loaders
    train_loader, val_loader, test_loader, loader_stats = initialize_loaders(data, config)
    
    # Initialize model
    model = initialize_model(config, model_config, device)
        
    # Initialize optimizer
    optimizer = initialize_optimizer(model, config)
        
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)

    # Initialize loss based on classification method and head
    criterion = initialize_criterion(config, loss_config)
    
    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0

    # Initialize statistics
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    # Create directory to save model and stats
    model_dir, stats_dir = create_model_stats_dir(config)
    
    # Training loop
    for epoch in range(config.epochs):

        #<----------Training---------->
        epoch_train_loss, epoch_train_acc = process_model(config, model, train_loader, criterion, device, "training", optimizer)

        #<----------Validation---------->
        epoch_val_loss, epoch_val_acc = process_model(config, model, val_loader, criterion, device, "evaluation", optimizer)
        
        print(f'Epoch: {epoch+1}, Train Loss: {epoch_train_loss}, Train Accuracy: {epoch_train_acc}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_acc}')

        #<----------Statistics---------->
        (train_loss.append((epoch, epoch_train_loss)), val_loss.append((epoch, epoch_val_loss)), train_acc.append((epoch, epoch_train_acc)), val_acc.append((epoch, epoch_val_acc)))

        # Weights & Biases Logging
        wandb_log(epoch, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc)
        
        #<----------Early Stopping---------->
        if early_stopping(epoch_val_loss, best_val_loss, counter, model, model_dir, config):
            break
        
        # Update learning rate
        scheduler.step()
        
        # Print learning rate
        print_learning_rate(config, optimizer)
    
    
    #<----------Testing---------->
    if config.test_ratio!=0:
        testing_and_logging(config, model, test_loader, criterion, device, optimizer, stats_dir)
        
    #<----------Save Statistics & Training Information---------->
    save_stats(train_loss, val_loss, train_acc, val_acc, stats_dir)
    
    #<----------Save Training Information---------->
    info_dict = get_wandb_info(config, model_config, loss_config, loader_stats)
    
    if config.test_ratio!=0:
        info_dict['Test examples'] = loader_stats["test_examples"]
        info_dict['Test batches'] = loader_stats["test_batches"]

    wandb.config.update(info_dict)
    save_final_json(info_dict, stats_dir)
    
    #<----------Complete Training Sessions---------->
    print("Training complete.")
    
    # Weights & Biases finish the experiment
    wandb.finish()