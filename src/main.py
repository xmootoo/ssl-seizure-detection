import sys
from train import train





if __name__ == '__main__':

    # Bash command line parameters
    
    # The path to .pt file of PairData objects list
    data_path = sys.argv[1]
    
    # The path to save the model
    logdir = sys.argv[2]
    
    # The path to save the training statistics
    patient_id = sys.argv[3]
    
    # The name of the model
    model_id = sys.argv[4]
    

    # PC
    data_path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_pyg\jh101\supervised\jh101_run1.pt"
    logdir = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\models"

    # Training parameters
    epochs = 200

    if model_id == "supervised":
        config = {
            "num_node_features": 9,
            "num_edge_features": 3,
            "hidden_channels": 64,
            "out_channels": 32,
            "dropout": 0.1,
        }
        data_size=1.0
        val_ratio=0.2
        test_ratio=0.1

    elif model_id == "relative_positioning":
        config = {
        "num_node_features": 9,
        "num_edge_features": 3,
        "hidden_channels": [64, 32, 64, 128, 256],
        }
        data_size=1.0
        val_ratio=0.2
        test_ratio=0
    elif model_id == "temporal_shuffling":
        pass

    train(data_path, logdir, patient_id, epochs, config, data_size, val_ratio, test_ratio, 
            batch_size=32, num_workers=4, lr=1e-3, weight_decay=1e-3, timing=True, model_id=model_id)

