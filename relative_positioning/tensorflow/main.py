import os
import pickle
from pathlib import Path

from tensorflow.keras.optimizers import Adam
from model import relative_positioning
from dataloader import get_data_loaders
from pseudodata import pseudo_data


# Data for jh101
path_pc = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_pseudolabeled/jh101_12s_7min.pkl"
data = pickle.load(open(path_pc, "rb"))



# Print within [[[A, NF, EF], [A', NF', EF']] Y] format
print(len(data[0]) == 2)
print(len(data[0][0]) == 2)
print(len(data[0][0][0]) == 3)

# [A, NF, EF]
print(data[0][0][0])

# [A', NF', EF']
print(data[0][0][1])

# [Y]
print(data[0][1][0])





# Parameters
fltrs_out = 64
l2_reg = 1e-3
lr = 1e-3
epochs = 100
val_size = 0.0
test_size = 0.2
batch_size = 32

# TODO: Not sure if we should select a different seed value
seed = 0


def train(data, fltrs_out, l2_reg, lr, epochs, batch_size, val_size, test_size, seed):
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size, val_size, test_size, seed)
    
    # Model
    model = relative_positioning(fltrs_out, l2_reg)
    
    # Optimization algorithm
    optimizer = Adam(lr)
    
    # Compile model
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC"])
    
    # Training
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            outs = model.train_on_batch(inputs, labels)
            for i, k in enumerate(model.metrics_names):
                if metrics is None:
                    metrics = {k: 0 for k in model.metrics_names}
                metrics[k] += outs[i] / len(train_loader)
        train_loss.append(metrics["loss"])
        train_acc.append(metrics["accuracy"])
        train_auc.append(metrics["auc"])
        train_f1_score.append(metrics["f1_score"])
    
    pass



    
    

# Model





# Optimizers
