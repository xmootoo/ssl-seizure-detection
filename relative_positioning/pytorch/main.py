import sys
from train import train




if __name__ == '__main__':
    
    # Training parameters
    epochs = 1
    data_size = 0.05
    train_ratio = 0.8
    batch_size = 32
    lr=1e-3
    timing=True

    # # Hard coded parameters
    # data_path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_pseudolabeled\relative_positioning\PyG\jh101_12s_7min_PairData.pt"
    # model_path = r"C:\Users\xmoot\Desktop\Models\PyTorch\ssl-seizure-detection\relative_positioning\jh101"
    # stats_path = r"C:\Users\xmoot\Desktop\Models\PyTorch\ssl-seizure-detection\relative_positioning\jh101"
    # model_name = r"\jh101_12s_7min_1epochs"
    # num_workers = 4

    # Bash command line parameters
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    stats_path = sys.argv[3]
    model_name = sys.argv[4]
    num_workers = int(sys.argv[5])

    train(data_path, model_path, stats_path, model_name, epochs, data_size, train_ratio, batch_size, num_workers,
          lr, timing)

