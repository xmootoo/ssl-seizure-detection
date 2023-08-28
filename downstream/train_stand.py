import os
import pickle
import time


from tensorflow.keras.optimizers import Adam
from model import gnn_standard
from preprocess import create_subfolder
from dataloader import dataloaders_torch
from evaluation import f1_score, training_curves, eval

start_time = time.time()


def run(data, fltrs_out, l2_reg, lr, epochs, batch_size, val_size, test_size, seed=0, es_patience=20,
        model_logdir=None, num_examples=50000, model_name=None):
    """
    Runnning the standard graph neural network model.

    Args:
        data (list): Graph pairs with pseudolabels of the form [[[A, NF, EF], [A', NF', EF']], Y].
        fltrs_out (int): Number of output filters for the GNN encoder.
        l2_reg (float): L2 regularization parameter.
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        batch_size (int): Batch size (powers of 2).
        val_size (float): Size of validation set (from 0 to 1).
        test_size (_type_): Size of test set (from 0 to 1)
        seed (int, optional): Random seed. Defaults to 0.
        es_patience (int, optional): Patience parameter. Defaults to 20.
        model_name (_type_, optional): Name of the model. Defaults to None.
        model_logdir (_type_, optional): Directory for saving the model. Defaults to None.
    """
    
    create_subfolder(model_logdir, model_name)
    save_path = model_logdir + "\\" + model_name
    
    # Get data loaders
    train_loader, val_loader, test_loader = dataloaders_torch(data, batch_size, val_size, test_size, seed, num_examples)
    
    # Model
    model = gnn_standard(fltrs_out, l2_reg)
    
    # Optimization algorithm
    optimizer = Adam(lr)
    
    # Compile model
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC", f1_score])
    
    # Metrics
    best_val_loss = 10000
    patience = es_patience
    train_loss = []
    train_acc = []
    train_auc = []
    train_f1_score = []
    val_loss = []
    val_acc = []
    val_auc = []
    val_f1_score = []
    saved_model = False
    
    for epoch in range(epochs):
        
        
        # ------------------------------ Training ------------------------------
        metrics = None
        
        # Adam optimization on batch
        for batch in train_loader:
            inputs, labels = batch
            
            # e.g., input[0] = [A, NF, EF]
            outs = model.train_on_batch(inputs, labels)
         
        # Training stats   
            for i, k in enumerate(model.metrics_names):
                if metrics is None:
                    metrics = {k: 0 for k in model.metrics_names}
                metrics[k] += outs[i] / len(train_loader)
        train_loss.append(metrics["loss"])
        train_acc.append(metrics["accuracy"])
        train_auc.append(metrics["auc"])
        train_f1_score.append(metrics["f1_score"])
        
        # Print loss
        print("Epoch: " + str(1 + epoch) + ". Loss: " + str(metrics["loss"]))
    
    
    # ------------------------------ Validation ------------------------------
        metrics_val = None
        for batch in val_loader:
            inputs, labels = batch
            
            # Convert to NumPy arrays
            inputs = [inp.numpy() for inp in inputs]         
            labels = labels.numpy()
            
            # Evaluate model
            outs = model.evaluate(inputs, labels)
            
            # Validation stats
            for i, k in enumerate(model.metrics_names):
                if metrics_val is None:
                    metrics_val = {k: 0 for k in model.metrics_names}
                metrics_val[k] += outs[i] / len(val_loader)
        val_loss.append(metrics_val["loss"])
        val_acc.append(metrics_val["accuracy"])
        val_auc.append(metrics_val["auc"])
        val_f1_score.append(metrics_val["f1_score"])
    
    
    # ------------------------------ Early stopping ------------------------------
        if metrics_val["loss"] < best_val_loss:
            best_val_loss = metrics_val["loss"]
            patience = es_patience
            model.save(os.path.join(save_path, model_name), save_format='tf')
            print(f"Model improved. Saving to {os.path.join(save_path, model_name + '.h5')}")
            saved_model = True
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break  # Exit the epoch loop
   
    
    # ------------------------------ Save best model ------------------------------
    model.load_weights(os.path.join(save_path, model_name + ".h5"))


    # ------------------------------ Save statistical measurements ------------------------------
    train_val_stats = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_auc': train_auc,
        'train_f1_score': train_f1_score,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_auc': val_auc,
        'val_f1_score': val_f1_score
    }
    pickle.dump(train_val_stats, open(os.path.join(save_path, "train_val_stats.pickle"), "wb"))

    
    # ------------------------------ Evaluation ------------------------------
    # Plot and save figures for training stats
    training_curves(train_loss, val_loss, train_acc, val_acc, save_path)
    
    # Save test stats
    test_stats, attns = eval(model, test_loader)
    pickle.dump(test_stats, open(os.path.join(model_logdir, "test_stats.pickle"), "wb"))
    

# PC Local paths
model_logdir = r"C:\Users\xmoot\Desktop\Models\TensorFlow\ssl-seizure-detection\models\supervised\standard"
data_path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_gr\jh101_grs.pickle"

# Load data
data = pickle.load(open(data_path, "rb"))

# Hyperparameters
fltrs_out=64
l2_reg=1e-3
lr=1e-3
epochs=1
batch_size=32
val_size=0.2
test_size=0.1
seed=0
es_patience=20
num_examples=5000
model_name = "jh101_" + str(epochs) + "epochs_" + str(num_examples) + "examples"

run(data=data, fltrs_out=fltrs_out, l2_reg=l2_reg, lr=lr, epochs=epochs, batch_size=batch_size, val_size=val_size, test_size=test_size,
    seed=seed, es_patience=es_patience, model_logdir=model_logdir, num_examples=num_examples, model_name=model_name)

# Takes approximately 1.5min to run 1000 examples
print("Process finished --- %s seconds ---" % (time.time() - start_time))