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

    # Training parameters
    epochs = 200
    val_ratio=0.2

    if model_id == "supervised":
        config = {
            "num_node_features": 9,
            "num_edge_features": 3,
            "hidden_channels": 64,
            "out_channels": 32,
            "dropout": 0.1,
        }
        data_size=1.0
        test_ratio=0.1

    elif model_id == "relative_positioning" or model_id == "temporal_shuffling":
        config = {
        "num_node_features": 9,
        "num_edge_features": 3,
        "hidden_channels": [64, 32, 64, 128, 256],
        }
        data_size=100000
        test_ratio=0

    train(data_path, logdir, patient_id, epochs, config, data_size, val_ratio, test_ratio, 
            batch_size=32, num_workers=4, lr=1e-3, weight_decay=1e-3, timing=True, model_id=model_id)

