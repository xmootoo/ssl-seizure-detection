import pandas as pd
import numpy as np
import torch
import random
import os
import copy
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split



def build_K_n(num_nodes):
    """
    Builds the edge_index for a complete graph K_n for num_nodes = n. 
    Credit: https://github.com/pyg-team/pytorch_geometric/issues/964

    Args:
        num_nodes (int): Number of nodes in the graph.
    
    Returns:
        E (numpy array): Edge index matrix of shape (2, num_edges) in PyG format.
    
    """
    # Initialize edge index matrix
    E = np.zeros((2, num_nodes * (num_nodes - 1)), dtype=np.int64)

    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes - 1):
            E[0, node * (num_nodes - 1) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node + 1, num_nodes)))
    E[1, :] = [item for sublist in neighbors for item in sublist]
    
    return E



def new_grs(data, type="preictal"):
    
    # Label data by Y = [Y_1, Y_2] where Y_1 is the binary encoding and Y_2 is the multiclass encoding.
    if type=="preictal":
        Y = [0, 0]
    elif type=="ictal":
        Y = [1, 1]
    elif type=="postictal":
        Y = [0, 2]
    
    new_grs = []
    for i in range(len(data)):
        # Node features
        NF_avg = data[i][1][1]
        NF_band = data[i][1][2]

        NF = np.concatenate((NF_avg, NF_band), axis=1)

        # Edge features
        EF_corr = data[i][2][1]
        EF_coh = data[i][2][2]
        EF_phase = data[i][2][3]

        EF = np.concatenate((EF_corr, EF_coh, EF_phase), axis=2)

        new_grs.append(([[NF, EF], Y]))
    
    return new_grs



def ef_to_edge_attr(edge_index, ef=None):
    """
    Stacks the weights of the adjacency matrix A as edge attributes to the edge_attr of a the graph.
    Note this ONLY works for complete graphs K_n as of now.

    Args:
        A (numpy array): Weighted adjacency matrix of shape (num_nodes, num_nodes).
        edge_index (numpy array): Edge indices array of shape (2, num_edges) in PyG format.
        edge_attr (numpy array): Existing edge features (if any), either of shape (num_nodes, num_nodes, num_edge_features) if given
                                 in functional connectivity matrix format. 

    Returns:
        edge_attr (numpy array): New edge features of shape (num_edges, num_edge_features) that follows the edge_index.
    """

    num_edges = edge_index.shape[1]
    num_nodes = ef.shape[1]
    num_edge_features = ef.shape[2]

    # Case 2: Edge features exist in FCN format. Convert adj matrix weights and edge features to edge_attr.
    edge_attr = np.zeros((num_edges, num_edge_features))
    for k, edge in enumerate(edge_index.T):
        i, j = edge
        edge_attr[k] = ef[i, j, :]

    return edge_attr



def adj_to_edge_attr(A, edge_index, edge_attr=None, mode=None):
    """
    Stacks the weights of the adjacency matrix A as edge attributes to the edge_attr of a the graph.
    Note this ONLY works for complete graphs K_n as of now.

    Args:
        A (numpy array): Weighted adjacency matrix of shape (num_nodes, num_nodes).
        edge_index (numpy array): Edge indices array of shape (2, num_edges) in PyG format.
        edge_attr (numpy array): Existing edge features (if any), either of shape (num_nodes, num_nodes, num_edge_features) if given
                                 in functional connectivity matrix format. Or of shape (num_edges, num_edge_features) if given in 
                                 PyG format. Defaults to None.
        mode (str): Format of edge_attr. Either "FCN" or "PyG". Defaults to None.

    Returns:
        edge_attr (numpy array): New edge features of shape (num_edges, num_edge_features + 1).
    """
    
    num_edges = edge_index.shape[1]
    num_nodes = A.shape[0]
    
    # Case 1: No edge features. Convert adjacency matrix weights to edge_attr.
    if mode is None:
        edge_attr_new = np.zeros((num_edges, 1))
        for k, edge in enumerate(edge_index.T):
            i, j = edge
            edge_attr_new[k] = A[i,j]

    # Case 2: Edge features exist in FCN format. Convert adj matrix weights and edge features to edge_attr.
    elif mode == "FCN":
        edge_attr_new = np.zeros((num_edges, 1 + edge_attr.shape[2]))
        for k, edge in enumerate(edge_index.T):
            i, j = edge
            edge_attr_new[k] = np.hstack((np.array([A[i, j]]), edge_attr[i, j, :]))
    
    # Case 3: Edge features exist in PyG format. Stack adj matrix weights on top of existing edge_attr.
    elif mode == "PyG":
        edge_attr_new = np.zeros((num_edges, 1))
        for k, edge in enumerate(edge_index.T):
            i, j = edge
            edge_attr_new[k] = A[i,j]
        edge_attr_new = np.hstack((edge_attr_new, edge_attr))
    else:
        return "Error: Mode not specified, must be either None, FCN, or PyG."

    return edge_attr_new



# Version of create_tensordata() but only for lists with entries [[NF, EF], Y] (no Adjacency matrix)
def create_tensordata_new(num_nodes, data_list, complete=True, save=True, logdir=None):
    """
    Converts the graph data from the pickle file containing the list of graph representations of with entries of the form [[NF, EF], Y]
    for numpy arrays NF, EF and float Y, to list of graph representations [[edge_index, x, edge_attr], y] for PyG format in torch tensors.
    
    args:
        num_nodes (int): Number of nodes in the graph.
        data_list (list): List of graph representations of the form [[A, NF, EF], Y] for numpy arrays A, NF, EF and float Y.
        complete (bool): Whether the graph is complete or not. Defaults to True.
    
    returns:
        pyg_data (list): List of graph representations of the form [[edge_index, x, edge_attr], y] for PyG format, where edge_index is a torch.long tensor of shape
                        (2, num_edges), x is a torch.float32 tensor of shape (num_nodes, num_node_features), edge_attr is a torch.float32 tensor of shape 
                        (num_edges, num_edge_features). 
    
    """
    pyg_data = []
    
    if complete:
        edge_index = build_K_n(num_nodes)
        edge_index = torch.from_numpy(edge_index).to(torch.long)

        for i, example in enumerate(data_list):

            # Parse data
            graph, y = example
            x, ef = graph
            
            # Conver to ef to edge_attr
            edge_attr = ef_to_edge_attr(edge_index, ef=ef)

            # Convert to tensors
            x = torch.from_numpy(x).to(torch.float32)
            # y = torch.tensor(y, dtype=torch.long)
            y = torch.tensor(y).view(1, -1).to(torch.long)
            edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
            
            pyg_data.append([[edge_index, x, edge_attr], y])

    if save:
        torch.save(pyg_data, logdir)
        
    return pyg_data



def create_tensordata(num_nodes, data_list, complete=True, save=True, logdir=None, mode=None):
    """
    Converts the graph data from the pickle file containing the list of graph representations of with entries of the form [[A, NF, EF], Y]
    for numpy arrays A, NF, EF and float Y, to list of graph representations [[edge_index, x, edge_attr], y] for PyG format in torch tensors.
    
    args:
        num_nodes (int): Number of nodes in the graph.
        data_list (list): List of graph representations of the form [[A, NF, EF], Y] for numpy arrays A, NF, EF and float Y.
        complete (bool): Whether the graph is complete or not. Defaults to True.
    
    returns:
        pyg_data (list): List of graph representations of the form [[edge_index, x, edge_attr], y] for PyG format, where edge_index is a torch.long tensor of shape
                        (2, num_edges), x is a torch.float32 tensor of shape (num_nodes, num_node_features), edge_attr is a torch.float32 tensor of shape 
                        (num_edges, num_edge_features). 
    
    """
    pyg_data = []
    
    if complete:
        edge_index = build_K_n(num_nodes)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        

        for i, example in enumerate(data_list):
    
            # Parse data
            graph, y = example
            A, x, _ = graph
            
            # Add adjacency matrix weights to edge attributes
            edge_attr = adj_to_edge_attr(A, edge_index, edge_attr=None, mode=mode)
            
            # Convert to tensors
            x = torch.from_numpy(x).to(torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
            
            pyg_data.append([[edge_index, x, edge_attr], y])

    if save:
        torch.save(pyg_data, logdir)
        
    return pyg_data



def graph_pairs(graph_reps, tau_pos=50, tau_neg=170, sample_ratio=1.0):
    """ 
    Creates unique sample pairs from a list of samples and their corresponding time indexes.

    Args:
        graph_reps (list of [graph_representation, Y]): Ordered list of graph representations each element is a list [graph_representaiton, Y] where
        Y is its label (ictal or nonictal). The index of graph_reps corresponds to the discrete time point of the entire iEEG recording, where one time point 
        is approx 0.12s.
        tau_pos: Positive context threshold.
        tau_neg: Negative context threshold.
        sample_ratio: Proportion of desired samples. Defaults to 1.0.

    Returns:
        graph_rep_pairs (list): List of graph representation pairs [gr_1, gr_2, Y], where Y corresponds to
        the pseudolabel of the pair.
    """
    n = len(graph_reps)

    sample_indices = random.sample(range(n), int(n * sample_ratio))

    graph_rep_pairs = []
    seen_pairs = set()

    # Get rid of original labels
    data = [graph_reps[i][0] for i in range(n)]

    pos_pairs = []
    neg_pairs = []

    for i in sample_indices:
        for j in sample_indices:
            if i >= j:
                continue
            diff = np.abs(i-j)

            if ((j, i) not in seen_pairs) and ((i, j) not in seen_pairs):
                if diff <= tau_pos:
                    pos_pairs.append([data[i], data[j], 1])
                elif diff > tau_neg:
                    neg_pairs.append([data[i], data[j], 0])
                seen_pairs.add((i, j))
    
    # Randomly shuffle both lists to ensure randomness
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    # Balance the dataset by using the minimum of the two class sizes
    min_size = min(len(pos_pairs), len(neg_pairs))

    # Trim down to the sample size
    pos_pairs = pos_pairs[:min_size]
    neg_pairs = neg_pairs[:min_size]

    # Concatenate the balanced data
    graph_rep_pairs = pos_pairs + neg_pairs

    # Shuffle the final dataset to ensure randomness
    random.shuffle(graph_rep_pairs)

    return graph_rep_pairs



def graph_triplets(graph_reps, tau_pos=50, tau_neg=170, sample_ratio=1.0):
    """
    Creates unique sample triplets from a list of samples and their corresponding time indexes.
    
    Args:
        graph_reps (list of [graph_representation, Y]): Ordered list of graph representations each element is a list [graph_representaiton, Y] where
        Y is its label (ictal or nonictal). The index of graph_reps corresponds to the discrete time point of the entire iEEG recording, where one time point 
        is approx 0.12s.
        tau_pos: Positive context threshold.
        tau_neg: Negative context threshold.
        sample_ratio: Proportion of desired samples. Defaults to 1.0.

    Returns:
        graph_rep_triplets (list): List of graph representation triplets [gr_1, gr_2, gr_3, Y], where Y corresponds to
        the pseudolabel of the triplet.
    """
    
    n = len(graph_reps)

    # Get rid of old labels
    data = [graph_reps[i][0] for i in range(n)]

    pos_triplets = []
    neg_triplets = []
    seen_triplets = set()

    # Only use a subset of indices, corresponding to our sample_ratio
    sample_indices = random.sample(range(n), int(n * sample_ratio))

    for t1 in sample_indices:
        for t3 in sample_indices:
            if t1 >= t3: # Ensuring that t1 < t3
                continue
            diff_pos = np.abs(t1 - t3) 

            if diff_pos <= tau_pos: # Positive context
                for t2 in sample_indices:
                    if t2 == t1 or t2 == t3:
                        continue

                    if (t1, t2, t3) in seen_triplets:
                        continue
                    
                    # Positive triplet
                    if t1 < t2 < t3:
                        pos_triplets.append([data[t1], data[t2], data[t3], 1])
                    
                    seen_triplets.add((t1, t2, t3)) # Seen triplet

                    # Negative triplet
                    # Compute the mid point and the distance of t2 from it
                    L = diff_pos / 2
                    midpoint = min(t1, t3) + L  # This handles both t1 < t3 and t1 > t3
                    diff_midpoint = np.abs(midpoint - t2)

                    # Check negative context and add triplet if so
                    if diff_midpoint > tau_neg / 2 and not (t1 < t2 < t3):
                        neg_triplets.append([data[t1], data[t2], data[t3], 0])
                        seen_triplets.add((t1, t2, t3))  # Remove redundant element
    

    # Balance the dataset by using the minimum of the two class sizes
    min_size = min(len(pos_triplets), len(neg_triplets))

    # Trim down to the sample size
    random.shuffle(pos_triplets)
    random.shuffle(neg_triplets)
    pos_triplets = pos_triplets[:min_size]
    neg_triplets = neg_triplets[:min_size]

    # Concatenate the balanced dataset
    graph_rep_triplets = pos_triplets + neg_triplets

    # Shuffle the final dataset to ensure randomness
    random.shuffle(graph_rep_triplets)

    return graph_rep_triplets


def gaussian_kernel(t1, t2, sigma):
    """
    Computes the Gaussian kernel value of two time points t1 and t2.

    Args:
        t1 (int): First time point.
        t2 (int): Second time point.
        sigma (float): Variance parameter of the Gaussian kernel.

    Returns:
        float: Gaussian kernel value of the two time points.
    """
    return np.exp(-np.square(t1 - t2) / (2 * np.square(sigma)))


def vicregt1_pairs(graph_reps, sigma=5, tau=0.68, sample_ratio=1.0):
    """ 
    Creates unique sample pairs from a list of samples and their corresponding time indexes.

    Args:
        graph_reps (list of [gr1, gr2, Y]): Ordered list of graph representations each element is a list [gr1, gr2, Y] where
        Y is its label (ictal or nonictal). The index of graph_reps corresponds to the discrete time point of the entire iEEG recording, where one time point 
        is approx 0.12s.
        sigma: The variance parameter of the Gaussian kernel.
        tau: The threshold parameter for the Gaussian kernel values.
        sample_ratio: Proportion of desired samples. Defaults to 1.0.

    Returns:
        graph_rep_pairs (list): List of graph representation pairs [gr1, gr2, y], where y=K(t1, t2) is the Guassian kernel value of the pair where each gr
                                is a list [x, edge_attr, edge_index] where x, edge_attr, and edge_index are the standard PyG tensors and y is a float32 tensor.
    """
    n = len(graph_reps)

    time_indices = random.sample(range(n), int(n * sample_ratio))

    graph_rep_pairs = []
    seen_pairs = set()

    # Get rid of original labels
    data = [graph_reps[i][0] for i in range(n)]

    for i in time_indices:
        for j in time_indices:
            
            # Compute kernel value
            y = gaussian_kernel(i, j, sigma)
            
            # Threshold and keep pair if it passes the threshold
            if y >= tau and i!=j and ((j, i) not in seen_pairs) and ((i, j) not in seen_pairs):
                graph_rep_pairs.append([data[i], data[j], torch.tensor(y, dtype=torch.float32)])
                seen_pairs.add((i, j))
                seen_pairs.add((j, i))
    
    # Randomly shuffle both lists to ensure randomness
    random.shuffle(graph_rep_pairs)

    return graph_rep_pairs




def cpc_tuples(data, K=5, N=5, P=1, data_size=100000):
    """
    Creates unique CPC tuples from an ordered list of data points for Contrastive Predictive Coding (CPC).
    Ideally you want to feed in the list of torch_geometric.data.Data objects saved in the .pt file.

    Args:
        data (list): Ordered list of data points, where each point is a representation of some observable quantity at a given time.
        K (int, optional, default=5): Number of past context data points to include in each tuple.
        N (int, optional, default=5): Number of negative samples to generate for each positive sample.
        P (int, optional, default=1): Number of future positive samples to include in each tuple.
        data_size (int, optional, default=100000): The number of CPC tuples to generate.

    Raises:
        ValueError: If there are not enough data points to generate P * N unique negative samples.

    Returns:
        cpc_samples (list of tuples): List of CPC tuples where each tuple contains three lists:
                                    The first list is of K past context data points,
                                    the second list is of P future positive examples,
                                    and the third list is of P * N future negative examples.
    """

    data = [[data[i], i] for i in range(len(data))]
    
    cpc_samples = []
    for i in range(data_size):
        # Ensure that there are enough unique samples for P * N negative samples
        if len(data) - (K + P) < P * N:
            raise ValueError("Not enough data to generate P * N unique negative samples.")

        # Select a starting index that allows for K context samples and P positive samples
        start_idx = random.randint(K, len(data) - P - 1)
        
        # Generate the context
        context = [data[start_idx - j][0] for j in range(K, 0, -1)]
        
        # Generate P positive samples
        x_positives = [data[start_idx + p][0] for p in range(P)]
        
        # Generate P * N negative samples
        x_negatives = random.sample([x for x, idx in data if idx < start_idx - K or idx > start_idx + P - 1], P * N)
        
        # Compile the sample
        sample = (context, x_positives, x_negatives)
        cpc_samples.append(sample)
    
    return cpc_samples


  

def pseudo_data(data, tau_pos=12 // 0.12, tau_neg=(9 * 60) // 0.12, stats=True, save=True, patientid="patient", 
                logdir=None, model="relative_positioning", sample_ratio=1.0, K=5, N=5, P=1, data_size=100000,
                sigma=5, tau=0.68):
    """
    Creates a pseudolabeled dataset of graph pairs, graph triplets, or CPC tuples from a list of graph representations.

    Args:
        data (list): Graph representations with labels of the form [[edge_index, x, edge_attr], y]
        tau_pos (int): Positive context threshold. Defaults to 6//0.12.
        tau_neg (int): Negative context threshold. Defaults to 50//0.12.
        stats (bool): Whether to display descriptive statistics on dataset or not. Defaults to True.
        save (bool): Whether to save as pickle file or not. Defaults to True.
        patientid (str): Patient identification code. Defaults to "patient".
        logdir (str): Directory to save the pickle file. Defaults to None.
        model (str): Model to use. Options: "relative_positioning", "temporal_shuffling", "CPC", "VICRegT1". Defaults to "relative_positioning".

    Returns:
        pairs (list): List of the form [[edge_index, x, edge_attr], [edge_index', x', edge_attr'], Y], where Y is the pseudolabel.
    """
    
    if logdir is None:
        logdir = ""
    
    if model == "relative_positioning":
        pairs = graph_pairs(data, tau_pos, tau_neg, sample_ratio)
        
        # Descriptive statistics    
        if stats:
            # Number of examples
            print("Number of examples: "  + str(len(pairs)))
            
            # Number of positive and negative examples
            df = pd.DataFrame(pairs, columns=['col1', 'col1', 'y'])
            counts = df['y'].value_counts()
            print(counts)
        
        # Save as a .pt file
        if save:
            torch.save(pairs, logdir + patientid + ".pt")
        
        return pairs
    
    elif model == "temporal_shuffling":
        triplets = graph_triplets(data, tau_pos, tau_neg, sample_ratio)
        
        # Descriptive statistics    
        if stats:
            # Number of examples
            print("Number of examples: "  + str(len(triplets)))
            
            # Number of positive and negative examples
            df = pd.DataFrame(triplets, columns=['col1', 'col2', 'col3', 'y'])
            counts = df['y'].value_counts()
            print(counts)
        
        # Save as a .pt file
        if save:
            torch.save(triplets, logdir + patientid + ".pt")
        
        return triplets
    
    elif model == "VICRegT1":
        pairs = vicregt1_pairs(data, sigma, tau, sample_ratio)
        
        if stats:
            print(f"Number of examples: {len(pairs)}")
        
        # Save as a .pt file
        if save:
            torch.save(pairs, logdir + patientid + ".pt")
            
        return pairs
    
    #TODO: Implement CPC.
    elif model == "CPC":
        pass
    

        
def convert_to_Data(data_list, save = True, logdir = None):
    """
    Converts a list of data entries of the form [[edge_index, x, edge_attr], y] to list of PyG Data objects.
    
    Args:
        data_list (list): A list of entries where each entry is of the form [[edge_index, x, edge_attr], y]. edge_index, x, edge_attr are 
        tensors representing graph components and y is a 1 dim tensor (label).
    
    Returns:
        Data_list (list): A list of PyG Data objects.
    
    """
    
    Data_list = []

    for entry in data_list:
        graph, y = entry
        edge_index, x, edge_attr = graph
        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
        Data_list.append(data)

    if save:
        torch.save(Data_list, logdir)
    
    return Data_list



class PairData(Data):
    """
    Creates the torch_geometric data object for a pair of graphs.
    
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        return super().__inc__(key, value, *args, **kwargs)



class TripletData(Data):
    """
    Creates the torch_geometric data object for a triplets of graphs.
    
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        if key == 'edge_index3':
            return self.x3.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class TupleData(Data):
    def __init__(self, M, graphs):
        super(TupleData, self).__init__()
        self.M = M
        assert len(graphs) == M, "Number of graphs must be equal to M."
        
        for idx, (edge_index, x, edge_attr) in enumerate(graphs):
            setattr(self, f'edge_index{idx+1}', edge_index)
            setattr(self, f'x{idx+1}', x)
            setattr(self, f'edge_attr{idx+1}', edge_attr)
            
    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            idx = int(key.split('_')[-1])  # assuming edge_index is followed by the index 1, 2, ...
            return getattr(self, f'x{idx}').size(0)
        return super().__inc__(key, value, *args, **kwargs)





def convert_to_PairData(data_list, save = True, logdir = None):
    """
    Converts a list of data entries of the form [[edge_index1, x1, edge_attr1] [edge_index2, x2, edge_attr2], y] to PyG Data objects.

    Args:
        data_list (list): A list of entries where each entry is of the form [[edge_index1, x1, edge_attr1] [edge_index2, x2, edge_attr2], y]. 
                            edge_index1, x1, edge_attr1, edge_index2, x2, edge_attr2 are tensors representing graph components and y is a 1 dim tensor (label).
    
    Returns:
        converted_data (list): A list of PyG PairData objects.
    """
    converted_data = []
    for entry in data_list:
        graph1, graph2, label = entry
        edge_index1, x1, edge_attr1 = graph1
        edge_index2, x2, edge_attr2 = graph2
        converted_data.append(PairData(x1=x1, edge_index1=edge_index1, edge_attr1=edge_attr1, 
                                       x2=x2, edge_index2=edge_index2, edge_attr2=edge_attr2, 
                                       y=label))
    
    if save:
        torch.save(converted_data, logdir)

    return converted_data


def convert_to_TripletData(data_list, save=True, logdir=None):
    """
    Converts a list of data entries of the form [graph1, graph2, graph3, y] to PyG Data objects.

    Args:
        data_list (list): A list of entries where each entry is of the form [[edge_index1, x1, edge_attr1] [edge_index2, x2, edge_attr2], [edge_index3, x3, edge_attr3], y]. 
                            edge_index_, x_, edge_attr_, are tensors representing graph components and y is a 1 dim tensor (label).
    
    Returns:
        converted_data (list): A list of PyG TripletData objects.
    """
    converted_data = []
    for entry in data_list:
        graph1, graph2, graph3, label = entry
        edge_index1, x1, edge_attr1 = graph1
        edge_index2, x2, edge_attr2 = graph2
        edge_index3, x3, edge_attr3 = graph3
        converted_data.append(TripletData(x1=x1, edge_index1=edge_index1, edge_attr1=edge_attr1, 
                                       x2=x2, edge_index2=edge_index2, edge_attr2=edge_attr2,
                                       x3=x3, edge_index3=edge_index3, edge_attr3=edge_attr3, 
                                       y=label))
    
    if save:
        torch.save(converted_data, logdir)

    return converted_data


def run_sorter(logdir, runtype="all"):
    """
    Returns the list of patients runs dependent on the settings.

    Args:
        logdir (str): Directory path (e.g., pyg_data/patient_id/model_id/), the folder containing the .pt runs.
        runtype (str): Specifies which runs to load. Options are "all", "combined", or "runx" where x is the run number. 
                       Defaults to "all".

    Returns:
        list or tensor: List of runs if runtype="all", single run otherwise.
    """
    if runtype == "combined":
        for run in os.listdir(logdir):
            if run.endswith("_combined.pt"):
                return torch.load(os.path.join(logdir, run))
    elif runtype == "all":
        all_runs = []
        for run in os.listdir(logdir):
            if not run.endswith("_combined.pt"):
                all_runs.append(torch.load(os.path.join(logdir, run)))
        return all_runs
    else:
        for run in os.listdir(logdir):
            if run.endswith(runtype + ".pt"):
                return torch.load(os.path.join(logdir, run))


def combiner(all_lists, desired_samples):
    """
    Combines multiple lists by randomly sampling from each, ensuring an almost equal contribution 
    from each list to meet a desired total number of samples.
    
    Args:
        all_lists (List[List[any]]): A list of lists to be combined.
        desired_samples (int): The total number of samples desired in the final list.
        
    Returns:
        List[any]: A list containing the sampled items from all input lists, shuffled.
    """
    # Check if sum of all list lengths is smaller than desired_samples
    total_length = sum(len(lst) for lst in all_lists)
    if total_length < desired_samples:
        final_list = [item for sublist in all_lists for item in sublist]
        random.shuffle(final_list)
        return final_list
    
    # Sort lists by length
    sorted_lists = sorted(all_lists, key=len)
    
    # Calculate initial quota
    Quota = desired_samples // len(all_lists)

    # Initialize an empty list to hold the final sampled elements
    final_list = []
    
    remaining_lists = len(sorted_lists)
    for lst in sorted_lists:
        remaining_lists -= 1  # Decrement the count of remaining lists
        random.shuffle(lst)  # Shuffle before sampling
        if len(lst) < Quota:
            final_list.extend(lst)
            if remaining_lists:  # Avoid division by zero
                Quota = (desired_samples - len(final_list)) // remaining_lists
        else:
            final_list.extend(random.sample(lst, Quota))

    # Shuffle final list to mix samples from different runs
    random.shuffle(final_list)
    
    return final_list



# def create_data_loaders(data, val_ratio=0.2, test_ratio=0.1, batch_size=32, num_workers=4, model_id="supervised", train_ratio=None):
def create_data_loaders(data, config):
    # Shuffle data
    """
    Create train and validation data loaders.

    Args:
        data (list): List of PyG Data, PairData, or TripletData objects.
        val_ratio (float): Proportion or fixed number of samples for validation. Defaults to 0.2.
        test_ratio (float): Proportion or fixed number of samples for testing. Defaults to 0.1. If no testing required set to 0.
        batch_size (int): Batch size. Defaults to 32.
        num_workers (int): Number of workers. Defaults to 4.
        model_id (str): Model to use. Either "supervised", "relative_positioning", or "temporal_shuffling". Defaults to "supervised".
        train_ratio (float or int, optional): Proportion or fixed number of samples for training. If None, uses the remaining samples after validation and testing.

    Returns:
        train_loader (PyG DataLoader): Training data loader.
        val_loader (PyG DataLoader): Validation data loader.
        test_loader (PyG DataLoader): Test data loader (optional). No test data loader is returned if test_ratio is set to 0.
    """
    
    # Take the random subset of the data
    n = len(data)
    indices = list(range(n))
    
    # Check for fixed sample sizes
    val_size = int(config.val_ratio) if config.val_ratio >= 1 else int(n * config.val_ratio)
    test_size = int(config.test_ratio) if config.test_ratio >= 1 else int(n * config.test_ratio)
    
    # If train_ratio is specified, compute train_size. Otherwise, compute based on remaining samples.
    if config.train_ratio:
        train_size = int(config.train_ratio) if config.train_ratio >= 1 else int(n * config.train_ratio)
    else:
        train_size = n - val_size - test_size

    # Ensure there's no overlap in sample sizes
    assert (train_size + val_size + test_size) <= n, "The sum of train, validation, and test sizes should not exceed the total number of samples."

    # Randomly sample indices for train, validation, and test sets without replacement
    all_indices = set(indices)
    val_indices = set(random.sample(all_indices, val_size))
    all_indices -= val_indices
    if config.test_ratio!=0:
        test_indices = set(random.sample(all_indices, test_size))
        all_indices -= test_indices
    if config.train_ratio:
        train_indices = set(random.sample(all_indices, train_size))
    else:
        train_indices = all_indices
    
    # Convert to lists
    train_idx = list(train_indices)
    val_idx = list(val_indices)
    test_idx = list(test_indices) if config.test_ratio != 0 else []

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx] if config.test_ratio != 0 else []
        
        
    # Create data loaders
    if config.model_id in {"supervised", "downstream1", "downstream2", "downstream3"}:
        train_loader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.num_workers)
        if config.test_ratio != 0:
            test_loader = DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers)
    elif config.model_id in {"relative_positioning", "VICRegT1"}:
        train_loader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, follow_batch=['x1', 'x2'])
        val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.num_workers, follow_batch=['x1', 'x2'])
        if config.test_ratio != 0:
            test_loader = DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers, follow_batch=['x1', 'x2'])
    elif config.model_id=="temporal_shuffling":
        train_loader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, follow_batch=['x1', 'x2', 'x3'])
        val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.num_workers, follow_batch=['x1', 'x2', 'x3'])
        if config.test_ratio != 0:    
            test_loader = DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers, follow_batch=['x1', 'x2', 'x3'])

    # Print Stats
    print(f"Total number of examples in dataset: {n}.")
    print(f"Total number of examples used: {len(indices)}.")
    print(f"Number of training examples: {len(train_data)}. Number of training batches: {len(train_loader)}.")
    print(f"Number of validation examples: {len(val_data)}. Number of validation batches: {len(val_loader)}.")
    if config.test_ratio != 0:
        test_data = [data[i] for i in test_idx]
        print(f"Number of test examples: {len(test_data)}. Number of test batches: {len(test_loader)}.")

    # Organize loaders and stats
    if config.test_ratio != 0:
        loaders = (train_loader, val_loader, test_loader)
        loader_stats = {"total_examples": len(data), "used_examples": len(indices), "train_examples": len(train_data), "val_examples": len(val_data), 
                        "test_examples": len(test_data), "train_batches": len(train_loader), "val_batches": len(val_loader), "test_batches": len(test_loader)}
    else:
        loaders = (train_loader, val_loader)
        loader_stats = {"total_examples": len(data), "used_examples": len(indices), "train_examples": len(train_data), "val_examples": len(val_data), "train_batches": len(train_loader), "val_batches": len(val_loader)}
    
    return loaders, loader_stats




def old_create_data_loaders(data, val_ratio=0.2, test_ratio=0.1, batch_size=32, num_workers=4, model_id="supervised"):
    # Shuffle data
    """
    Create train and validation data loaders.

    Args:
        data (list): List of PyG Data, PairData, or TripletData objects.
        val_ratio (float): Proportion of the data to be used for validation. Defaults to 0.2.
        test_ratio (float): Proportion of the data to be used for testing. Defaults to 0.1. Set to 0 if no testing data is required.
        batch_size (int): Batch size. Defaults to 32.
        num_workers (int): Number of workers. Defaults to 4.
        model_id (str): Model to use. Either "supervised", "relative_positioning", or "temporal_shuffling". Defaults to "supervised".

    Returns:
        train_loader (PyG DataLoader): Training data loader.
        val_loader (PyG DataLoader): Validation data loader.
        test_loader (PyG DataLoader): Test data loader (optional). No test data loader is returned if test_ratio is set to 0.
    """
    
    # Take the random subset of the data
    n = len(data)
    indices = list(range(n))
    
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, shuffle=True)
    train_data, val_data = [data[i] for i in train_idx], [data[i] for i in val_idx]
    if test_ratio != 0:
        train_idx, test_idx = train_test_split(train_idx, test_size=test_ratio / (1 - val_ratio), shuffle=True)
        test_data = [data[i] for i in test_idx]

    # Create data loaders
    if model_id=="supervised" or model_id=="downstream1" or model_id=="downstream2":
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
        if test_ratio != 0:
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    elif model_id=="relative_positioning":
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2'])
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2'])
        if test_ratio != 0:    
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2'])
    elif model_id=="temporal_shuffling":
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2', 'x3'])
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2', 'x3'])
        if test_ratio != 0:    
            test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2', 'x3'])

    # Print Stats
    print(f"Total number of examples in dataset: {n}.")
    print(f"Total number of examples used: {len(indices)}.")
    print(f"Number of training examples: {len(train_data)}. Number of training batches: {len(train_loader)}.")
    print(f"Number of validation examples: {len(val_data)}. Number of validation batches: {len(val_loader)}.")
    if test_ratio != 0:
        test_data = [data[i] for i in test_idx]
        print(f"Number of test examples: {len(test_data)}. Number of test batches: {len(test_loader)}.")

    # Organize loaders and stats
    if test_ratio != 0:
        loaders = (train_loader, val_loader, test_loader)
        loader_stats = {"total_examples": len(data), "used_examples": len(indices), "train_examples": len(train_data), "val_examples": len(val_data), 
                        "test_examples": len(test_data), "train_batches": len(train_loader), "val_batches": len(val_loader), "test_batches": len(test_loader)}
    else:
        loaders = (train_loader, val_loader)
        loader_stats = {"total_examples": len(data), "used_examples": len(indices), "train_examples": len(train_data), "val_examples": len(val_data), "train_batches": len(train_loader), "val_batches": len(val_loader)}
    
    return loaders, loader_stats
        


def extract_layers(model_path, model_dict_path, transfer_id):
    """
    Extracts pretrained layers of a model.

    Args:
        model_path (str): Path to the model.
        model_dict_path (str): Path to the model state dictionary.
        transfer_id (str): Model to use. Either "relative_positioning" or "temporal_shuffling".
    
    Returns:
        pretrained_layers (list): List of pretrained layers.
    
    """
    # Load model
    model = torch.load(model_path)
    model.eval()

    # Load state dictionary
    model_dict = torch.load(model_dict_path)
    
    if transfer_id=="relative_positioning" or transfer_id=="temporal_shuffling":
        EdgeMLP_pretrained = copy.deepcopy(model.embedder.edge_mlp)
        NNConv_pretrained = copy.deepcopy(model.embedder.conv1)
        GATConv_pretrained = copy.deepcopy(model.embedder.conv2)
        pretrained_layers = {"edge_mlp": EdgeMLP_pretrained, "conv1": NNConv_pretrained, "conv2": GATConv_pretrained}
    
    elif transfer_id=="VICRegT1":
        edge_mlp = copy.deepcopy(model.embedder.edge_mlp)
        conv1 = copy.deepcopy(model.embedder.conv1)
        conv2 = copy.deepcopy(model.embedder.conv2)
        conv3 = copy.deepcopy(model.embedder.conv3)
        bn_graph1 = copy.deepcopy(model.embedder.bn_graph1)
        bn_graph2 = copy.deepcopy(model.embedder.bn_graph2)
        bn_graph3 = copy.deepcopy(model.embedder.bn_graph3)
        pretrained_layers = {"edge_mlp": edge_mlp,
                        "conv1": conv1,
                        "conv2": conv2,
                        "conv3": conv3,
                        "bn_graph1": bn_graph1,
                        "bn_graph2": bn_graph2,
                        "bn_graph3": bn_graph3,
                        }
    
    return pretrained_layers
