import pickle
import pandas as pd
import numpy as np
import torch


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



def convert_to_tensor(data):
    """
    Converts a list of data entries of the form [[A, X, E], Y] to PyTorch tensors.
    
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
        tf_graphs = [torch.tensor(tensor) for tensor in graphs]
        tf_label = torch.tensor(label)
        converted_data.append([tf_graphs, tf_label])
    return converted_data


    
def adj_to_pyg(A, weighted = True):
    """ 
    Converts an adjacency matrix to the edge index and edge weight that pytorch_geometric uses to track edges.

    Args:
        A (torch.Tensor): Shape: (num_nodes, num_nodes), the binary adjaceny matrix.
    
    Returns:
        edge_index (torch.Tensor): Shape: (2, num_edges), the edge index matrix. Each column represents a directed edge.
        edge_weight (torch.Tensor): Shape: (num_edges,), the edge weight vector. Each entry represents the weight of the corresponding edge in edge_index.
        weighted (bool): Whether the adjacency matrix is weighted or not.
    """

    # Find the indices where the adjacency matrix is nonzero
    # These represent the directed edges
    i, j = np.where(A != 0)

    # Create the edge_index tensor with shape (2, num_edges)
    edge_index = torch.tensor([i, j], dtype=torch.long)

    # Create the edge_weight tensor with shape (num_edges,)
    # Using the values from the adjacency matrix at the corresponding indices
    if weighted:
        edge_weight = torch.tensor([A[i[k], j[k]] for k in range(len(i))], dtype=torch.float)
        return edge_index, edge_weight
    else:
        return edge_index

    





# Test case
adj_matrix = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
edge_index, edge_weight = convert_adjacency_matrix(adj_matrix)
print(edge_index)
print(edge_weight)