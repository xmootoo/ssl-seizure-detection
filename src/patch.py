import os
import pickle
from preprocess import new_grs, create_tensordata_new, convert_to_Data, pseudo_data, convert_to_PairData, convert_to_TripletData

def patch(graphrep_dir=None,  logdir=None, file_name="", num_electrodes=107, tau_pos=12//0.12, tau_neg=60//0.12, 
          model="supervised", stats=True, save=True, sample_ratio=1.0):
    """
    Preprocesses and convert various types of graph representations (GRs) to PyTorch Geometric data format.

    The function takes in pickled graph representation data found in '/User/projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements', 
    preprocesses it according to the specified model type, and then converts the data into a PyTorch Geometric-friendly format. The function
    supports supervised, relative positioning, and temporal shuffling models. It saves the converted data to the given log directory.

    Args:
        graphrep_dir (tuple): Paths to the preictal, ictal, and postictal pickle files. 
                              Format: (path_preictal, path_ictal, path_postictal).
        logdir (str, optional): Directory where the processed PyTorch Geometric data will be saved.
        file_name (str, optional): Name of the saved PyTorch Geometric data file (no extension, e.g., "jh101").
        num_electrodes (int, optional): Number of electrodes in the graph representation. Default is 107.
        tau_pos (float, optional): Positive time constant for the relative positioning or temporal shuffling model. 
                                   Default is 12//0.12.
        tau_neg (float, optional): Negative time constant for the relative positioning or temporal shuffling model.
                                   Default is 60//0.12.
        model (str, optional): Type of model for which the graph data is being prepared. 
                               Options: "supervised", "relative_positioning", "temporal_shuffling". Default is "supervised".
        stats (bool, optional): Whether to display statistics about the pseudolabeled data. Default is True.
        save (bool, optional): Whether to save the processed PyTorch Geometric data. Default is True.
        sample_ratio (float, optional): Ratio of the samples to be considered for processing. 
                                        Useful for sub-sampling. Default is 1.0.
    
    Returns:
        list of PyTorch Geometric Data: If model is "supervised", returns a list of PyTorch Geometric Data objects.
        list of PyTorch Geometric PairData: If model is "relative_positioning", returns a list of PyTorch Geometric PairData objects.
        list of PyTorch Geometric TripletData: If model is "temporal_shuffling", returns a list of PyTorch Geometric TripletData objects.
    """
       
    # Create the save directory with .pt extension
    if save:
        logdir = os.path.join(logdir, file_name + ".pt")

    # Load pickle data of standard graph representations (GRs) corresponding to Alan's dictionary
    path_preictal, path_ictal, path_postictal = graphrep_dir

    data_preictal = pickle.load(open(path_preictal, 'rb'))
    data_ictal = pickle.load(open(path_ictal, 'rb'))
    data_postictal = pickle.load(open(path_postictal, 'rb'))

    # Select graph representation (GR) type from Alan's dictionary of GRs
    new_data_preictal = new_grs(data_preictal, type="preictal", mode="binary")
    new_data_ictal = new_grs(data_ictal, type="ictal", mode="binary")
    new_data_postictal = new_grs(data_postictal, type="postictal", mode="binary")

    # Concatenate all data temporally
    new_data = new_data_preictal + new_data_ictal + new_data_postictal

    # Convert standard graph representations to Pytorch Geometric data
    pyg_grs = create_tensordata_new(num_nodes=num_electrodes, data_list=new_data, complete=True, save=False, logdir=None)
    
    if model == "supervised":
        Data_list = convert_to_Data(pyg_grs, save=save, logdir=logdir)
        return Data_list
    
    if model == "relative_positioning":
        pdata = pseudo_data(pyg_grs, tau_pos=tau_pos, tau_neg=tau_neg, stats=stats, save=False, patientid="", 
                            logdir=None, model="relative_positioning", sample_ratio=sample_ratio)
        Pair_Data = convert_to_PairData(pdata, save=save, logdir=logdir)
        return Pair_Data

    if model == "temporal_shuffling":
        pdata = pseudo_data(pyg_grs, tau_pos=tau_pos, tau_neg=tau_neg, stats=stats, save=False, patientid="", 
                            logdir=None, model="temporal_shuffling", sample_ratio=sample_ratio)
        Triplet_Data = convert_to_TripletData(pdata, save=save, logdir=logdir)
        return Triplet_Data
            



    
    
  