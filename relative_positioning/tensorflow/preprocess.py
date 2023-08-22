import pickle
import pandas as pd
import numpy as np
import tensorflow as tf


def graph_pairs(graph_reps, tau_pos = 50, tau_neg = 170):
    """ 
    Creates unique sample pairs from a list of samples and their corresponding time indexes.

    Args:
        graph_reps (list of [[A, NF, EF], Y]): Ordered list of graph representations each element is a list [[A, NF, EF], Y] where
        A is the adj matrix, NF are the node features, EF are the edge features, and Y is its label (ictal or nonictal). The index
        of graph_reps corresponds to the discrete time point of the entire iEEG recording, where one time point is approx 0.12s.
        tau_pos: Positive context threshold.
        tau_neg: Negative context threshold.

    Returns:
        graph_rep_pairs (numpy array): List of graph representation pairs [[[A, NF, EF], [A', NF', EF']], Y], where Y corresponds to
        the pseudolabel of the pair.
    """

    n = len(graph_reps)
    graph_rep_pairs = []
    done_tasks = set()
    
    # Get rid of the real labels
    data = [graph_reps[i][0] for i in range(n)]

    # Create pairs and pseudolabels
    for i in range(n):
        for j in range(n):
            
            # Time distance between pairs
            diff = np.abs(i-j)
            
            # Check if pair entry unique and each pair is unique up to permutation
            if (i != j) & ((j, i) not in done_tasks):
                
                # Check if pair is within the positive or negative context
                if diff <= tau_pos:
                    graph_rep_pairs.append([[data[i], data[j]], 1])
                elif diff > tau_neg:
                    graph_rep_pairs.append([[data[i], data[j]], 0])
                done_tasks.add((i, j))
        
    return graph_rep_pairs


def adj(A, thres):
    """Converts functional connectivity matrix to binary adjacency matrix.

    Args:
        A (numpy array): Functional connectivity matrix.
        thres (float): Threshold value.
    """
    
    
    n = A.shape[0]
    x = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if A[i,j] > thres:
                x[i,j] = 1
    
    return x



def pseudo_data(data, tau_pos = 6 // 0.12, tau_neg = 50 // 0.12, mode = "weighted", stats = True, save = True, patientid = "patient", logdir = None):
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
        with open(logdir + patientid + ".pkl", "wb") as f:
            pickle.dump(x, f)
    
    return x



def convert_to_tf_tensors(data):
    """
    Converts a list of data entries of the form [[A, X, E], Y] to TensorFlow tensors.
    
    Args:
        data (list): A list of entries where each entry is of the form [[A, X, E], Y].
                     A, X, E, and Y are NumPy arrays representing graph components and labels.
    
    Returns:
        list: A list of entries with the same structure as the input data, but with each
              NumPy array replaced by a TensorFlow tensor.
              
    Example:
        data = [[[A, X, E], Y], ...]
        tf_data = convert_to_tf_tensors(data)
    """
    converted_data = []
    for entry in data:
        graphs, label = entry
        tf_graphs = [tf.convert_to_tensor(tensor) for tensor in graphs]
        tf_label = tf.convert_to_tensor(label)
        converted_data.append([tf_graphs, tf_label])
    return converted_data



    
# # Test case for adj
# # Functional connectivity matrix and threshold
# A = np.array([[0.5, -1], [0.2, 0.4]])
# thres = 0.3
# print(adj(A, thres))

# # Test case for graph_pairs
# # Graph representations with labels
# graph_reps = [
#     [["adj1", "nf1", "ef1"], 1],
#     [["adj2", "nf2", "ef2"], 0],
#     [["adj3", "nf3", "ef3"], 1],
#     [["adj4", "nf4", "ef4"], 0]
# ]

# # Positive and negative context thresholds
# tau_pos = 1
# tau_neg = 2

# # Call the function
# pairs = graph_pairs(graph_reps, tau_pos, tau_neg)

# # Print the pairs
# for pair in pairs:
#     print(pair)
