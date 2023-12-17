import sys
from train import train

    
if __name__ == '__main__':

    # The path to .pt file of Data, PairData, or TripletData objects list
    data_path = sys.argv[1]
    
    # Path to save the model, statistics, and training information
    logdir = sys.argv[2]
    
    # Patient identifier
    patient_id = sys.argv[3]
    
    # Model identifier
    model_id = sys.argv[4]

    # Date and time identifier
    datetime_id = sys.argv[5]

    # Run identifier
    run_type = sys.argv[6]

    # Task (binary, multiclass)
    classify = str(sys.argv[7])

    # Train, val, test split. Must formatted as "train,val,test" in the command line.
    split = [float(x) for x in sys.argv[8].split(",")]
    if len(split) == 3:
        train_ratio, val_ratio, test_ratio = split[0], split[1], split[2]
    elif len(split) == 2:
        val_ratio, test_ratio = split[0], split[1]
        train_ratio=None
    
    # Num of epochs
    epochs = int(sys.argv[9])
    
    # Project id
    project_id = str(sys.argv[10])
    
    # Transfer learning (optional arguments)
    if len(sys.argv) > 11:
        model_path = str(sys.argv[11])
        model_dict_path = str(sys.argv[12])
        transfer_id = str(sys.argv[13])
        frozen = bool(int(sys.argv[14])) # convert 0 or 1 to False or True
    else:
        model_path=None
        model_dict_path=None
        transfer_id=None
        frozen=None

    # Node feature dimension configuration (some patients have less node features)
    if patient_id in {"ummc003", "ummc004", "ummc006"}:
        num_node_features = 7
    elif patient_id in {"ummc001", "ummc002"}:
        num_node_features = 8
    else:
        num_node_features = 9

    
    # Training parameters
    if model_id == "supervised":
        config = {
            "num_node_features": num_node_features,
            "num_edge_features": 3,
            "hidden_channels": 64,
            "out_channels": 32,
            "dropout": 0.1,
        }
        data_size=1.0
    
    elif model_id == "relative_positioning" or model_id == "temporal_shuffling":
        config = {
        "num_node_features": num_node_features,
        "num_edge_features": 3,
        "hidden_channels": [64, 32, 64, 128, 256],
        }
        data_size=115000
    
    elif model_id == "downstream1":
        config = {
        "hidden_channels": [64, 64, 32],
        "dropout": 0.1,
        }
        data_size=1.0
        
    elif model_id == "downstream2":
        config = {
        "hidden_channels": 32,
        "dropout": 0.1,
        }
        data_size=1.0

    elif model_id == "VICRegT1":
        
        # Model and loss parameters
        config = {
            "num_node_features": num_node_features,
            "num_edge_features": 3,
            "hidden_channels": [64, 128, 128, 512, 512, 512],
            "batch_norm": True,
            "dropout": True,
            "p": 0.1,
            "classify": None,
            "head": None,
            }
        
        # Data size
        loss_config = {
            "loss_coeffs": (25, 25, 1), 
            "y_scale": True, 
            "gamma": 1, 
            "epsilon": 1e-4,
            }
        lr=0.01
        patience=float("inf")
        data_size=100000
        batch_size=256
        weight_decay=1e-6
        dropout=True
        eta_min=1e-5

    
    train(data_path=data_path, logdir=logdir, patient_id=patient_id, epochs=epochs, config=config, data_size=data_size, val_ratio=val_ratio, test_ratio=test_ratio, 
          batch_size=batch_size, num_workers=8, lr=lr, weight_decay=weight_decay, model_id=model_id, timing=True, classify=config["classify"], head=config["head"], 
          dropout=dropout, datetime_id=datetime_id, run_type=run_type, frozen=frozen, model_path=model_path, model_dict_path=model_dict_path, transfer_id=transfer_id, train_ratio=train_ratio, 
          loss_config=loss_config, project_id=project_id, patience=patience, eta_min=eta_min)