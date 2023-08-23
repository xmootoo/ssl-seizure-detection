import pickle
import sys
from train_tf import run

# Model, stats, and pseudolabeled data directories from command line
# model_logdir = sys.argv[1]
# stats_logdir = sys.argv[2]
# pseudodata_path = sys.argv[3]


# PC Local paths
model_logdir = r"C:\Users\xmoot\Desktop\Models\TensorFlow\ssl-seizure-detection\models"
stats_logdir = r"C:\Users\xmoot\Desktop\Models\TensorFlow\ssl-seizure-detection\stats"
pseudodata_path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_pseudolabeled\relative_positioning\jh101_12s_7min_np.pkl"



# Load data
data = pickle.load(open(pseudodata_path, "rb"))

# Hyperparameters
fltrs_out=64
l2_reg=1e-3
lr=1e-3
epochs=10
batch_size=32
val_size=0.2
test_size=0
seed=0
es_patience=20

run(data=data, fltrs_out=fltrs_out, l2_reg=l2_reg, lr=lr, epochs=epochs, batch_size=batch_size, val_size=val_size, test_size=test_size,
    seed=seed, es_patience=es_patience, stats_logdir=stats_logdir, model_logdir=model_logdir)
