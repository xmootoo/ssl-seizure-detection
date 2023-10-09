import os
import pickle
from preprocess import new_grs, create_tensordata_new, convert_to_Data, pseudo_data, convert_to_PairData, convert_to_TripletData

def patch(graphrep_dir=None,  logdir=None, file_name=None, num_electrodes=107, tau_pos=12//0.12, tau_neg=(7 * 60)//0.12, 
          model="supervised", stats=True, sample_ratio=1.0):
       
    # Create the save directory with .pt extension
    logdir = os.path.join(logdir, file_name + ".pt")

    # Load pickle data of standard graph representations
    path_preictal, path_ictal, path_postictal = graphrep_dir

    data_preictal = pickle.load(open(path_preictal, 'rb'))
    data_ictal = pickle.load(open(path_ictal, 'rb'))
    data_postictal = pickle.load(open(path_postictal, 'rb'))

    # Select new data for each class
    new_data_preictal = new_grs(data_preictal, type="preictal", mode="binary")
    new_data_ictal = new_grs(data_ictal, type="ictal", mode="binary")
    new_data_postictal = new_grs(data_postictal, type="postictal", mode="binary")

    # Concatenate all data temporally
    new_data = new_data_preictal + new_data_ictal + new_data_postictal

    # Convert standard graph representations to Pytorch Geometric data
    pyg_grs = create_tensordata_new(num_nodes=107, data_list=new_data, complete=True, save=False, logdir=None)
    
    if model == "supervised":
        Data_list = convert_to_Data(pyg_grs, save=True, logdir=logdir)
        return Data_list

    if model == "relative_positioning":
        pdata = pseudo_data(pyg_grs, tau_pos=tau_pos, tau_neg=tau_neg, stats=stats, save=False, patientid="", 
                            logdir=None, model="relative_positioning", sample_ratio=sample_ratio)
        Pair_Data = convert_to_PairData(pdata, save=True, logdir=logdir)
        return Pair_Data

    if model == "temporal_shuffling":
        pdata = pseudo_data(pyg_grs, tau_pos=tau_pos, tau_neg=tau_neg, stats=stats, save=False, patientid="", 
                            logdir=None, model="temporal_shuffling", sample_ratio=sample_ratio)
        Triplet_Data = convert_to_TripletData(pdata, save=True, logdir=logdir)
        return Triplet_Data
            



    
    
  