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
        sample_ratio (int, optional): Proportion of samples to be used in relative positioning or temporal shuffling. Defaults to 1.0.
    
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

    with open(path_preictal, 'rb') as f:
        data_preictal = pickle.load(f)
    with open(path_ictal, 'rb') as f:
        data_ictal = pickle.load(f)
    with open(path_postictal, 'rb') as f:
        data_postictal = pickle.load(f)

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


def full_patcher(user="xmootoo", patient_dir=None, logdir=None, num_electrodes=107, tau_pos=12//0.12, tau_neg=60//0.12, 
             model="supervised", stats=True, save=True, sample_ratio=1.0):
    """
    
    Automates the patch() across all patients.

    """

    # String IDs for each patient
    patient_ids = {
        "jh101", "jh103", "jh108",
        "pt01", "pt2", "pt3", "pt6", "pt7", "pt8", "pt10", "pt11", "pt12", 
        "pt13", "pt14", "pt15", "pt16",
        "umf001", "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", 
        "ummc006", "ummc007", "ummc008", "ummc009"
    }

    # Assign directory of patient folders
    if patient_dir==None:
        directory = os.path.join("/home", user, "projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements")
    else:
        directory=patient_dir
    
    try:
        # Iterate through each patient in the directory
        for patient in os.listdir(directory):
            if patient not in patient_ids:
                continue
            
            # Create a patient folder in the log directory
            patient_logdir = os.path.join(logdir, patient)
            if not os.path.exists(patient_logdir):
                os.makedirs(patient_logdir)
            
            # Create model-specific directory within patient_logdir
            model_logdir = os.path.join(patient_logdir, model)
            if not os.path.exists(model_logdir):
                os.makedirs(model_logdir)

            # Form a path for the patient folder
            full_path = os.path.join(directory, patient)
            if os.path.isdir(full_path):
                
                # Count how many runs there are
                runs = 0
                for file in os.listdir(full_path):
                    if file.startswith("preictal"):
                        runs += 1
                
                # Iterate through all runs in the patient folder
                for i in range(1, runs+1):
                    path_preictal = os.path.join(full_path, f"preictal_{i}.pickle")
                    path_ictal = os.path.join(full_path, f"ictal_{i}.pickle")
                    path_postictal = os.path.join(full_path, f"postictal_{i}.pickle")

                    # Create the graphrep_dir
                    graphrep_dir = (path_preictal, path_ictal, path_postictal)

                    # Supervised
                    if os.path.exists(path_preictal) and os.path.exists(path_ictal) and os.path.exists(path_postictal):
                        if model == "supervised":
                            file_name = patient + "_run" + str(i)
                        if model == "relative_positioning" or model == "temporal_shuffling":
                            file_name = patient + "_run" + str(i) + "_" + str(int(tau_pos * 0.12)) + "s" + "_" + str(int(tau_neg * 0.12)) + "_" + str(sample_ratio) + "sr"
                        patched_data = patch(graphrep_dir=graphrep_dir,  logdir=model_logdir, file_name=file_name, num_electrodes=num_electrodes, tau_pos=tau_pos, 
                                             tau_neg=tau_neg, model=model, stats=stats, save=save, sample_ratio=sample_ratio)
                        
    except Exception as e:
        print(f"An error occurred: {e}")




def single_patient_patcher(user="xmootoo", patient_dir=None, patient=None, logdir=None, num_electrodes=107, tau_pos=12//0.12, tau_neg=60//0.12, 
             model="supervised", stats=True, save=True, sample_ratio=1.0):
    """
    Automates the patch() function for a single patient.
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
        if not os.path.exists(patient_logdir):
            os.makedirs(patient_logdir)
        
        # Create model-specific directory within patient_logdir
        model_logdir = os.path.join(patient_logdir, model)
        if not os.path.exists(model_logdir):
            os.makedirs(model_logdir)

        # Form a path for the patient folder
        full_path = os.path.join(directory, patient)
        if os.path.isdir(full_path):
            
            # Count how many runs there are
            runs = 0
            for file in os.listdir(full_path):
                if file.startswith("preictal"):
                    runs += 1
            
            # Iterate through all runs in the patient folder
            for i in range(1, runs+1):
                path_preictal = os.path.join(full_path, f"preictal_{i}.pickle")
                path_ictal = os.path.join(full_path, f"ictal_{i}.pickle")
                path_postictal = os.path.join(full_path, f"postictal_{i}.pickle")

                # Create the graphrep_dir
                graphrep_dir = (path_preictal, path_ictal, path_postictal)

                if os.path.exists(path_preictal) and os.path.exists(path_ictal) and os.path.exists(path_postictal):
                    file_name = patient + "_run" + str(i)
                    if model == "relative_positioning" or model == "temporal_shuffling":
                        file_name = file_name + "_" + str(int(tau_pos * 0.12)) + "s" + "_" + str(int(tau_neg * 0.12)) + "_" + str(sample_ratio) + "sr"
                    
                    patched_data = patch(graphrep_dir=graphrep_dir,  logdir=model_logdir, file_name=file_name, num_electrodes=num_electrodes, tau_pos=tau_pos, 
                                         tau_neg=tau_neg, model=model, stats=stats, save=save, sample_ratio=sample_ratio)
                        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import sys
    patient_dir = str(sys.argv[1])
    patient = str(sys.argv[2])
    logdir = str(sys.argv[3])
    model = str(sys.arg[4])
    sample_ratio = float(sys.argv[5])

    single_patient_patcher(user="xmootoo", patient_dir=patient_dir, patient=patient, logdir=logdir, num_electrodes=107, tau_pos=12//0.12, tau_neg=90//0.12, 
                           model=model, stats=True, save=True, sample_ratio=sample_ratio)






    
    
  