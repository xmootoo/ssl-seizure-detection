import pickle
from preprocess import adj, graph_pairs
import pandas as pd


# File path (PC) for subject jh101
path = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_gr/jh101_grs.pickle"

# File path (Macbook) for subject jh101
# path = "/Users/xaviermootoo/Documents/Data/SSL-seizure-detection/pickle/jh101_grs.pickle"


# Load graph representations list of [[A, NF, EF], Y]]
data = pickle.load(open(path, "rb"))


tau_pos = 6 // 0.12
tau_neg = 50 // 0.12
pseudo_data = graph_pairs(data, tau_pos, tau_neg)


def pseudo_data(data, tau_pos = 6 // 0.12, tau_neg = 50 // 0.12, mode = "weighted", stats = True, save = True, patientid = "patient"):
    """
    Creates a pseudolabeled dataset of graph pairs.
    

    Args:
        data (list): Graph representations with labels of the form [[A, NF, EF], Y]
        tau_pos (int): Positive context threshold. Defaults to 6//0.12.
        tau_neg (int): Negative context threshold. Defaults to 50//0.12.
        mode (str): Mode of adjacency matrix. Defaults to "weighted".
        stats (bool): Whether to display descriptive statistics on dataset or not. Defaults to True.
        save (bool): Whether to save as pickle file or not. Defaults to True.
        patientid (str): Patient identification code. Defaults to "patient".

    Returns:
        x (list): List of the form [[[A,NF,EF], [A',NF',EF']], Y]
    """
    
    
    # Create list of  graph pairs, with entries of the form [[A, NF, EF], [A', NF', EF'], Y]
    # A and A' depend on the mode selected
    
    # Weighted adjacency matrix mode (FCN)
    if mode == "weighted":
        x = graph_pairs(data, tau_pos, tau_neg)
    
    # Binary adjacency matrix mode
    if mode == "adj":
        for i in range(len(data)):
            data[i][0][0] = adj(data[i][0][0], 0.3)
        
        x = graph_pairs(data, tau_pos, tau_neg)
    
    # Descriptive statistics    
    if stats == True:
        # Number of examples
        print("Number of examples: "  + str(len(x)))
        
        # Number of positive and negative examples
        df = pd.DataFrame(x, columns=['col1', 'y'])
        counts = df['y'].value_counts()
        print(counts)
    
    # Save as a pickle file
    if save == True:
        folder_path = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_pseudolabeled/"
        with open(folder_path + patientid + ".pkl", "wb") as f:
            pickle.dump(x, f)
    
    return x

    
# For subject jh101, we achieve roughly a 50/50 split of positive negative samples with tau_pos = 12s and tau_neg = 7min
pdata = pseudo_data(data, tau_pos = 12 // 0.12, tau_neg = (7 * 60) // 0.12, mode = "weighted", stats = True, save = True, patientid = "jh101_12s_7min")




# # Test case
# path_pc = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_pseudolabeled/jh101_12s_7min.pkl"
# data = pickle.load(open(path_pc, "rb"))



# # Print within [[[A, NF, EF], [A', NF', EF']] Y] format
# print(len(data[0]) == 2)
# print(len(data[0][0]) == 2)
# print(len(data[0][0][0]) == 3)

# # [A, NF, EF]
# print(data[0][0][0])

# # [A', NF', EF']
# print(data[0][0][1])

# # [Y]
# print(data[0][1][0])
