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
    
    # Transfer learning (optional arguments)
    if len(sys.argv) > 9:
        model_path = str(sys.argv[9])
        model_dict_path = str(sys.argv[10])
        transfer_id = str(sys.argv[11])
        frozen = bool(int(sys.argv[12])) # convert 0 or 1 to False or True
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
    epochs=200
    
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
            "num_node_features": 9,
            "num_edge_features": 3,
            "hidden_channels": [64, 128, 128, 512, 512, 512],
            "batch_norm": True,
            }
        
        # Data size
        loss_config = {
            "loss_coeffs":(25, 25, 1), 
            "y_scale":True, 
            "gamma":1, 
            "epsilon":1e-4,
            }


    train(data_path, logdir, patient_id, epochs, config, data_size, val_ratio, test_ratio, 
        batch_size=32, num_workers=4, lr=1e-3, weight_decay=1e-3, model_id=model_id, timing=True, 
        classify=classify, head="linear", dropout=True, datetime_id=datetime_id, run_type=run_type, 
        frozen=frozen, model_path=model_path, model_dict_path=model_dict_path, transfer_id=transfer_id,
        train_ratio=train_ratio)