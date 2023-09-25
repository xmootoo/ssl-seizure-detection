import pickle
import os
from preprocess import create_tensordata, convert_to_Data, pseudo_data, convert_to_PairData, convert_to_TripletData

def patch(graphrep_dir=None, logdir=None, file_name=None, mode="binary", model="supervised", stats=True):
  
  # Create the save directory with .pt extension
  logdir = os.join(logdir, file_name + ".pt")

  # Load pickle data of standard graph representations
  pickle_data = pickle.load(open(graphrep_dir, "rb"))

  # Convert standard graph representations to Pytorch Geometric data
  if mode == "binary":
    pyg_grs = create_tensordata(num_nodes=107, data_list=pickle_data, complete=True, save=False, logdir=None, mode="binary")
  elif mode == "multi":
    pyg_grs = create_tensordata(num_nodes=107, data_list=pickle_data, complete=True, save=False, logdir=None, mode="multi")
  
  # Select which model to use
  if model == "supervised":
    Data_list = convert_to_Data(pyg_grs, save=True, logdir=logdir)
    return Data_list
  
  if model == "relative_positioning":
    pdata = pseudo_data(pyg_grs, tau_pos = 12 // 0.12, tau_neg = (7 * 60) // 0.12, stats=stats, save=False, patientid="", 
                        logdir=None, model="relative_positioning")
    Pair_Data = convert_to_PairData(pdata, save=True, logdir=logdir)
    return Pair_Data
  
  if model == "temporal_shuffling":
    pass
  




