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



def adj_to_edge_attr(A, edge_index, edge_attr=None):
    """
    Stacks the weights of the adjacency matrix A as edge attributes to the edge_attr of a the graph.

    Args:
        A (numpy array): Adjacency matrix.
        edge_index (numpy array): Edge indices of shape (2, num_edges).
        edge_attr (numpy array): Existing edge features of shape (num_edges, num_edge_features).

    Returns:
        edge_attr (numpy array): New edge features of shape (num_edges, num_edge_features + 1).
    """
    
    num_edges = edge_index.shape[1]
    edge_attr_new = np.zeros((num_edges, 1))
    
    # Filling up the new attribute with values from the adjacency matrix
    for k, edge in enumerate(edge_index.T):
        i, j = edge
        edge_attr_new[k] = A[i,j]
        
    # If edge_attr is None, the new attribute becomes the edge_attr
    if edge_attr is None:
        edge_attr = edge_attr_new
    else:
        # If edge_attr exists, concatenate the new attribute to the existing edge attributes
        edge_attr = np.hstack((edge_attr, edge_attr_new))

    return edge_attr



def create_tensordata(num_nodes, data_list, complete=True, save=True, logdir=None, mode="binary"):
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
            edge_attr = adj_to_edge_attr(A, edge_index)
            
            # Convert to tensors
            x = torch.from_numpy(x).to(torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
            
            pyg_data.append([[edge_index, x, edge_attr], y])

    if save:
        torch.save(pyg_data, logdir)
        
    return pyg_data



def graph_pairs(graph_reps, tau_pos = 50, tau_neg = 170):
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
                    graph_rep_pairs.append([data[i], data[j], 1])
                elif diff > tau_neg:
                    graph_rep_pairs.append([data[i], data[j], 0])
                done_tasks.add((i, j))
        
    return graph_rep_pairs


#TODO: Fix this function, it's giving triplets with duplicate entries.
def graph_triplets_new(graph_reps, tau_pos=50, tau_neg=170, data_size=1.0):
    n = len(graph_reps)
    
    # Create a list containing the graph representation and the index
    data = [[graph_reps[i][0], i] for i in range(n)]
    
    # Shuffle the data
    random.shuffle(data)
    
    # Trim the data based on data_size
    trim_size = int(n * data_size)
    data = data[:trim_size]
    
    graph_rep_triplets = []
    seen_triplets = set()
    
    for t1_index in range(len(data)):
        for t2_index in range(len(data)):
            for t3_index in range(len(data)):
                t1 = data[t1_index][1]
                t2 = data[t2_index][1]
                t3 = data[t3_index][1]
                diff_pos = np.abs(t1 - t3)
                
                if diff_pos <= tau_pos and t2 != t1 and t2 != t3 and ((t1, t2, t3) not in seen_triplets):
                        if (t1 < t2 < t3) or (t3 < t2 < t1):
                            graph_rep_triplets.append([data[t1_index][0], data[t2_index][0], data[t3_index][0], 1])
                            seen_triplets.add((t1, t2, t3))
                            
                        M = diff_pos / 2
                        diff_third = np.abs(M - t2)
                        
                        if diff_third > tau_neg // 2:
                            graph_rep_triplets.append([data[t1_index][0], data[t2_index][0], data[t3_index][0], 0])
                            graph_rep_triplets.append([data[t3_index][0], data[t2_index][0], data[t1_index][0], 0])
                            seen_triplets.add((t1, t2, t3))
                            seen_triplets.add((t3, t2, t1))
                            
    return graph_rep_triplets





def graph_triplets(graph_reps, tau_pos=50, tau_neg=170, data_size=1.0):
    n = len(graph_reps)
    data = [graph_reps[i][0] for i in range(n)]
    
    graph_rep_triplets = []
    seen_triplets = set()
    
    
    for t1 in range(n):
        for t3 in range(t1 + 1, n):  # Ensuring t1 < t3 to leverage symmetry
            diff_pos = np.abs(t1 - t3)
            
            if diff_pos <= tau_pos:
                for t2 in range(n):
                    if t2 != t1 and t2 != t3:
                        if (t1, t2, t3) not in seen_triplets:
                            if (t1 < t2 < t3) or (t3 < t2 < t1):
                                graph_rep_triplets.append([data[t1], data[t2], data[t3], 1])
                                seen_triplets.add((t1, t2, t3))
                                
                            M = diff_pos / 2
                            diff_third = np.abs(M - t2)
                            
                            if diff_third > tau_neg // 2:
                                graph_rep_triplets.append([data[t1], data[t2], data[t3], 0])
                                graph_rep_triplets.append([data[t3], data[t2], data[t1], 0])  # Adding the mirrored triplet
                                seen_triplets.add((t1, t2, t3))
                                seen_triplets.add((t3, t2, t1))
                            
    return graph_rep_triplets





def pseudo_data(data, tau_pos = 12 // 0.12, tau_neg = (7 * 60) // 0.12, stats = True, save = True, patientid = "patient", 
                logdir = None, model = "relative_positioning", data_size = 1.0):
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
        pairs = graph_pairs(data, tau_pos, tau_neg)
        
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
        triplets = graph_triplets(data, tau_pos, tau_neg)
        
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
    """Converts a list of data entries of the form [[edge_index, x, edge_attr], y] to list of PyG Data objects.
    
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
    """Converts a list of data entries of the form [[edge_index1, x1, edge_attr1] [edge_index2, x2, edge_attr2], y] to PyG Data objects.

    Args:
        data_list (list): A list of entries where each entry is of the form [[edge_index1, x1, edge_attr1] [edge_index2, x2, edge_attr2], y]. 
                            edge_index1, x1, edge_attr1, edge_index2, x2, edge_attr2 are tensors representing graph components and y is a 1 dim tensor (label).
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
    """Converts a list of data entries of the form [graph1, graph2, graph3, y] to PyG Data objects.

    Args:
        data_list (list): A list of entries where each entry is of the form [[edge_index1, x1, edge_attr1] [edge_index2, x2, edge_attr2], [edge_index3, x3, edge_attr3], y]. 
                            edge_index_, x_, edge_attr_, are tensors representing graph components and y is a 1 dim tensor (label).
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
