import pickle
from train import run




# Run for jh101
# Load pseudolabel dataset with tensorflow
path_pc = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_pseudolabeled/jh101_12s_7min_np.pkl"
data = pickle.load(open(path_pc, "rb"))


# Arguments
model_logdir = r"C:\Users\xmoot\Desktop\Models\TensorFlow\ssl-seizure-detection\models"
stats_logdir = r"C:\Users\xmoot\Desktop\Models\TensorFlow\ssl-seizure-detection\stats"

run(data=data, fltrs_out=64, l2_reg=1e-3, lr=1e-3, epochs=5, batch_size=32, val_size=0.2, test_size=0,
    seed=0, es_patience=20, stats_logdir=stats_logdir, model_logdir=model_logdir)
