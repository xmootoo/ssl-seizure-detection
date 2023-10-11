import pandas as pd
import numpy as np
import torch
import random
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader



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



def new_grs(data, type="preictal", mode="binary"):
    
    # Encode labels
    # Binary encoding
    if mode == "binary":
        if type == "ictal":
            Y = 1
        if type == "preictal":
            Y = 0
        if type == "postictal":
            Y = 0

    # Multiclass encoding
    if mode == "multiclass":
        if type == "ictal":
            Y = 0
        if type == "preictal":
            Y = 1
        if type == "postictal":
            Y = 2

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
            y = torch.tensor(y, dtype=torch.long)
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

    Returns:
        graph_rep_pairs (list): List of graph representation pairs [gr_1, gr_2, Y], where Y corresponds to
        the pseudolabel of the pair.
    """

    n = len(graph_reps)
    graph_rep_pairs = []
    done_tasks = set()

    # Get rid of original labels
    data = [graph_reps[i][0] for i in range(n)]

    pos_pairs = []
    neg_pairs = []

    for i in range(n):
        for j in range(n):
            diff = np.abs(i-j)

            if (i != j) & ((j, i) not in done_tasks):
                if diff <= tau_pos:
                    pos_pairs.append([data[i], data[j], 1])
                elif diff > tau_neg:
                    neg_pairs.append([data[i], data[j], 0])
                done_tasks.add((i, j))

    # Randomly shuffle both lists to ensure randomness
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)

    # Balance the dataset by using the minimum of the two class sizes
    min_size = min(len(pos_pairs), len(neg_pairs))

    # Apply the sample_ratio to scale down
    sample_size = int(min_size * sample_ratio)
    
    # Trim down to the sample size
    pos_pairs = pos_pairs[:sample_size]
    neg_pairs = neg_pairs[:sample_size]

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
        sample_ratio: Proportion of the psuedodata to be sampled from the entire dataset. Defaults to 1.0.

    Returns:
        graph_rep_triplets (list): List of graph representation triplets [gr_1, gr_2, gr_3, Y], where Y corresponds to
        the pseudolabel of the triplet.
    """
    
    n = len(graph_reps)
    data = [graph_reps[i][0] for i in range(n)]

    pos_triplets = []
    neg_triplets = []
    seen_triplets = set()

    for t1 in range(n):
        for t3 in random.sample(range(t1 + 1, n), min(int((n - t1 - 1) * sample_ratio), n - t1 - 1)):
            diff_pos = np.abs(t1 - t3)

            available_t2 = [x for x in range(n) if x != t1 and x != t3]
            for t2 in random.sample(available_t2, min(int(n * sample_ratio), len(available_t2))):
                if (t1, t2, t3) not in seen_triplets:

                    M = diff_pos / 2
                    diff_third = np.abs(M - t2)

                    if (t1 < t2 < t3) or (t3 < t2 < t1):
                        pos_triplets.append([data[t1], data[t2], data[t3], 1])
                        seen_triplets.add((t1, t2, t3))

                    if diff_third > tau_neg // 2:
                        if not ((t1 < t2 < t3) or (t3 < t2 < t1)):
                            neg_triplets.append([data[t1], data[t2], data[t3], 0])
                            seen_triplets.add((t1, t2, t3))

    # Randomly shuffle lists
    random.shuffle(pos_triplets)
    random.shuffle(neg_triplets)

    # Balance the dataset by using the minimum of the two class sizes
    min_size = min(len(pos_triplets), len(neg_triplets))

    # Apply the sample_ratio to scale down
    sample_size = int(min_size * sample_ratio)

    # Trim down to the sample size
    pos_triplets = pos_triplets[:sample_size]
    neg_triplets = neg_triplets[:sample_size]

    # Concatenate the balanced dataset
    graph_rep_triplets = pos_triplets + neg_triplets

    # Shuffle the final dataset to ensure randomness
    random.shuffle(graph_rep_triplets)

    return graph_rep_triplets



def pseudo_data(data, tau_pos = 12 // 0.12, tau_neg = (7 * 60) // 0.12, stats = True, save = True, patientid = "patient", 
                logdir = None, model = "relative_positioning", sample_ratio = 1.0):
    """
    Creates a pseudolabeled dataset of graph pairs.
    

    Args:
        data (list): Graph representations with labels of the form [[edge_index, x, edge_attr], y]
        tau_pos (int): Positive context threshold. Defaults to 6//0.12.
        tau_neg (int): Negative context threshold. Defaults to 50//0.12.
        stats (bool): Whether to display descriptive statistics on dataset or not. Defaults to True.
        save (bool): Whether to save as pickle file or not. Defaults to True.
        patientid (str): Patient identification code. Defaults to "patient".

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
        
        # Save as a pickle file
        if save:
            torch.save(pairs, logdir + patientid + ".pt")
        
        return pairs
    
    if model == "temporal_shuffling":
        triplets = graph_triplets(data, tau_pos, tau_neg, sample_ratio)
        
        # Descriptive statistics    
        if stats:
            # Number of examples
            print("Number of examples: "  + str(len(triplets)))
            
            # Number of positive and negative examples
            df = pd.DataFrame(triplets, columns=['col1', 'col2', 'col3', 'y'])
            counts = df['y'].value_counts()
            print(counts)
        
        if save:
            torch.save(triplets, logdir + patientid + ".pt")
        
        return triplets


        
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


def convert_to_TripletData(data_list, save = True, logdir = None):
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



def create_data_loaders(data, data_size=1.0, train_ratio=0.8, batch_size=32, num_workers=4):
    # Shuffle data
    """
    Create train and validation data loaders.

    Parameters:
    - data (list): The dataset
    - val_split (float): Fraction of data to go into validation set
    - batch_size (int): Size of mini-batches
    - num_workers (int): Number of worker threads to use with DataLoader

    Returns:
    - train_loader, val_loader: DataLoader instances for training and validation data
    """
    
    # Shuffle data
    random.shuffle(data)
    
    # Take the subset of the data
    n = len(data)
    n_subset = int(n * data_size)
    data_subset = data[:n_subset]

    # Calculate the size of the training and validation sets
    train_size = int(len(data_subset) * train_ratio)
    val_size = len(data_subset) - train_size

    # Split the data
    train_data, val_data = random_split(data_subset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2'])
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, follow_batch=['x1', 'x2'])

    return train_loader, val_loader



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









# # Old version of graph triplets
# def graph_triplets(graph_reps, tau_pos=50, tau_neg=170, sample_ratio=1.0):
#     """
#     Creates unique sample triplets from a list of samples and their corresponding time indexes.
    
#     Args:
#         graph_reps (list of [graph_representation, Y]): Ordered list of graph representations each element is a list [graph_representaiton, Y] where
#         Y is its label (ictal or nonictal). The index of graph_reps corresponds to the discrete time point of the entire iEEG recording, where one time point 
#         is approx 0.12s.
#         tau_pos: Positive context threshold.
#         tau_neg: Negative context threshold.
#         sample_ratio: Proportion of the psuedodata to be sampled from the entire dataset. Defaults to 1.0.

#     Returns:
#         graph_rep_triplets (list): List of graph representation triplets [gr_1, gr_2, gr_3, Y], where Y corresponds to
#         the pseudolabel of the triplet.

    
#     """
#     n = len(graph_reps)
#     data = [graph_reps[i][0] for i in range(n)]
    
#     graph_rep_triplets = []
#     seen_triplets = set()

#     for t1 in range(n):
#         for t3 in random.sample(range(t1 + 1, n), min(int((n - t1 - 1) * sample_ratio), n - t1 - 1)):
#             diff_pos = np.abs(t1 - t3)
            
#             if diff_pos <= tau_pos:
#                 available_t2 = [x for x in range(n) if x != t1 and x != t3]
#                 for t2 in random.sample(available_t2, min(int(n * sample_ratio), len(available_t2))):
#                     if (t1, t2, t3) not in seen_triplets:
                        
#                         M = diff_pos / 2
#                         diff_third = np.abs(M - t2)

#                         # Checking the condition for label=1
#                         if (t1 < t2 < t3) or (t3 < t2 < t1):
#                             graph_rep_triplets.append([data[t1], data[t2], data[t3], 1])
#                             seen_triplets.add((t1, t2, t3))
                        
#                         # Checking the condition for label=0
#                         if diff_third > tau_neg // 2:
#                             if not ((t1 < t2 < t3) or (t3 < t2 < t1)):
#                                 graph_rep_triplets.append([data[t1], data[t2], data[t3], 0])
#                                 seen_triplets.add((t1, t2, t3))
                            
#     return graph_rep_triplets