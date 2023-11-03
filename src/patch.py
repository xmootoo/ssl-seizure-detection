import os
import pickle
import torch
import re
from preprocess import new_grs, create_tensordata_new, convert_to_Data, pseudo_data, convert_to_PairData, convert_to_TripletData, cpc_tuples

def patch(graphrep_dir=None,  logdir=None, file_name="", tau_pos=12//0.12, tau_neg=60//0.12, 
          model="supervised", stats=True, save=True, sample_ratio=1.0, K=5, N=5, P=1, data_size=100000):
    """
    Preprocesses and convert various types of graph representations (GRs) to PyTorch Geometric data format.

    The function takes in pickled graph representation data found in '/User/projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements', 
    preprocesses it according to the specified model type, and then converts the data into a PyTorch Geometric-friendly format. The function
    supports supervised, relative positioning, and temporal shuffling models. It saves the converted data to the given log directory.

    Args:
        graphrep_dir (tuple): Paths to the preictal, ictal, and postictal pickle files. 
                              Format: (path_preictal, path_ictal, path_postictal). If model_id == "CPC", graphrep_dir is the path to the supervised data
                              in the form of a .pt file, i.e. the list of Data objects.
        logdir (str, optional): Directory where the processed PyTorch Geometric data will be saved.
        file_name (str, optional): Name of the saved PyTorch Geometric data file (no extension, e.g., "jh101").
        tau_pos (float, optional): Positive time constant for the relative positioning or temporal shuffling model. 
                                   Default is 12//0.12.
        tau_neg (float, optional): Negative time constant for the relative positioning or temporal shuffling model.
                                   Default is 60//0.12.
        model (str, optional): Type of model for which the graph data is being prepared. 
                               Options: "supervised", "relative_positioning", "temporal_shuffling". Default is "supervised".
        stats (bool, optional): Whether to display statistics about the pseudolabeled data. Default is True.
        save (bool, optional): Whether to save the processed PyTorch Geometric data. Default is True.
        sample_ratio (int, optional): Proportion of samples to be used in relative positioning or temporal shuffling. Defaults to 1.0.
    
    Returns:
        list of PyTorch Geometric Data: If model is "supervised", returns a list of PyTorch Geometric Data objects.
        list of PyTorch Geometric PairData: If model is "relative_positioning", returns a list of PyTorch Geometric PairData objects.
        list of PyTorch Geometric TripletData: If model is "temporal_shuffling", returns a list of PyTorch Geometric TripletData objects.
    """
       
    # Create the save directory with .pt extension
    if save:
        logdir = os.path.join(logdir, file_name + ".pt")

    if model == "CPC":
        data = torch.load(graphrep_dir)
        cpc_data = cpc_tuples(data, K=K, N=N, P=P, data_size=data_size)
        return cpc_data
    
    else:
        # Load pickle data of standard graph representations (GRs) corresponding to Alan's dictionary
        path_preictal, path_ictal, path_postictal = graphrep_dir

        with open(path_preictal, 'rb') as f:
            data_preictal = pickle.load(f)
        with open(path_ictal, 'rb') as f:
            data_ictal = pickle.load(f)
        with open(path_postictal, 'rb') as f:
            data_postictal = pickle.load(f)
        
        # Select graph representation (GR) type from Alan's dictionary of GRs
        new_data_preictal = new_grs(data_preictal, type="preictal")
        new_data_ictal = new_grs(data_ictal, type="ictal")
        new_data_postictal = new_grs(data_postictal, type="postictal")

        # Concatenate all data temporally
        new_data = new_data_preictal + new_data_ictal + new_data_postictal

        # Get number of electrodes
        num_electrodes = new_data[0][0][0].shape[0]
        
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



def single_patient_patcher(user="xmootoo", patient_dir=None, patient=None, logdir=None, tau_pos=12//0.12, tau_neg=60//0.12, 
             model="supervised", stats=True, save=True, sample_ratio=1.0, K=5, N=5, P=1, data_size=100000):
    """
    
    Automates the patch() function for a single patient.
    
    Args:
        user (str, optional): Compute Canada username of the user. Default is "xmootoo".
        patient_dir (str, optional): Directory where the patient folders are located. Default is None.
        patient (str, optional): Patient identifier. Default is None.
        logdir (str, optional): Directory where the processed PyTorch Geometric data will be saved. Default is None.
        tau_pos (float, optional): Positive time constant for the relative positioning or temporal shuffling model.
                                      Default is 12//0.12.
        tau_neg (float, optional): Negative time constant for the relative positioning or temporal shuffling model.
                                        Default is 60//0.12.
        model (str, optional): Type of PyTorch model to be used. Options: "supervised", "relative_positioning", "temporal_shuffling", "CPC".
                                 Default is "supervised".
        stats (bool, optional): Whether to display statistics about the pseudolabeled data. Default is True.
        save (bool, optional): Whether to save the processed PyTorch Geometric data. Default is True.
        sample_ratio (int, optional): Proportion of samples to be used in relative positioning or temporal shuffling. Defaults to 1.0.
    
    Saves:
        list of PyTorch Geometric Data: If model is "supervised", returns a list of PyTorch Geometric Data objects.
        list of PyTorch Geometric PairData: If model is "relative_positioning", returns a list of PyTorch Geometric PairData objects.
        list of PyTorch Geometric TripletData: If model is "temporal_shuffling", returns a list of PyTorch Geometric TripletData objects.
    """
    # Assign directory of patient folders
    if patient is None:
        print("Please provide a patient identifier.")
        return

    if patient_dir==None:
        directory = os.path.join("/home", user, "projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements")
    else:
        directory=patient_dir

    try:
        # Create a patient folder in the log directory
        patient_logdir = os.path.join(logdir, patient)
        os.makedirs(patient_logdir, exist_ok=True)
        
        # Create model-specific directory within patient_logdir
        model_logdir = os.path.join(patient_logdir, model)
        os.makedirs(model_logdir, exist_ok=True)

        if model == "relative_positioning" or model == "temporal_shuffling":
            sp_model_logdir = os.path.join(model_logdir, str(int(tau_pos * 0.12)) + "s_" + str(int(tau_neg * 0.12)) + "s_" + str(sample_ratio) + "sr")
            os.makedirs(sp_model_logdir, exist_ok=True)
        
        if model == "CPC":
            sp_model_logdir = os.path.join(model_logdir, str(K) + "K_" + str(N) + "N_" + str(P) + "P_" + str(data_size) + "ds")
            os.makedirs(sp_model_logdir, exist_ok=True)
            graphrep_dir = os.path.join(directory, patient, "supervised", patient + "_combined.pt")
            patched_data = patch(graphrep_dir=graphrep_dir, logdir=sp_model_logdir, file_name=file_name, tau_pos=tau_pos, tau_neg=tau_neg, 
                                 model=model, stats=stats, save=False, sample_ratio=sample_ratio, K=K, N=N, P=P, data_size=data_size)
            if save:
                torch.save(patched_data, os.path.join(sp_model_logdir, patient + ".pt"))
                
        
        # Form a path for the patient folder
        full_path = os.path.join(directory, patient)
        if os.path.isdir(full_path):
            
            # Count how many runs there are
            runs = []
            
            # Loop through each file in the directory
            for file in os.listdir(full_path):
                
                # Check if the file starts with "preictal"
                if file.startswith("preictal"):
                    
                    # Extract the run number using regex
                    match = re.search(r"preictal_(\d+)", file)
                    if match:
                        run_number = int(match.group(1))
                        runs.append(run_number)
            
            runs = sorted(runs)
            
            # Iterate through all runs in the patient folder
            patched_data_list = []
            
            for i in runs:
                path_preictal = os.path.join(full_path, f"preictal_{i}.pickle")
                path_ictal = os.path.join(full_path, f"ictal_{i}.pickle")
                path_postictal = os.path.join(full_path, f"postictal_{i}.pickle")

                # Create the graphrep_dir
                graphrep_dir = (path_preictal, path_ictal, path_postictal)

                if os.path.exists(path_preictal) and os.path.exists(path_ictal) and os.path.exists(path_postictal):
                    file_name = patient + "_run" + str(i)
                    if model == "relative_positioning" or model == "temporal_shuffling":
                        save_dir = sp_model_logdir
                    else:
                        save_dir = model_logdir
                        
                    patched_data = patch(graphrep_dir=graphrep_dir,  logdir=save_dir, file_name=file_name, tau_pos=tau_pos, 
                                         tau_neg=tau_neg, model=model, stats=stats, save=save, sample_ratio=sample_ratio)
                    
                    patched_data_list.append(patched_data)
            
            # Save combined data using all runs
            for data in patched_data_list:
                combined_data = []
                for data in patched_data_list:
                    combined_data += data

            file_name = patient + "_combined"
            
            # Save concatenated_data using torch.save
            if save:
                torch.save(combined_data, os.path.join(save_dir, file_name + ".pt"))
            return
            
                
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    import sys
    patient_dir = str(sys.argv[1])
    patient = str(sys.argv[2])
    logdir = str(sys.argv[3])
    model = str(sys.argv[4])
    
    if model == "CPC":
        K = int(sys.argv[5])
        N = int(sys.argv[6])
        P = int(sys.argv[7])
        data_size = int(sys.argv[8])
        sample_ratio=1.0
    else:
        tau_pos = float(sys.argv[5])
        tau_neg = float(sys.argv[6])
        sample_ratio = float(sys.argv[7])
        K, N, P, data_size = 0, 0, 0, 0

    single_patient_patcher(user="xmootoo", patient_dir=patient_dir, patient=patient, logdir=logdir, tau_pos=tau_pos, tau_neg=tau_neg, 
             model=model, stats=True, save=True, sample_ratio=sample_ratio K=K, N=N, P=P, data_size=data_size)





    
    
  