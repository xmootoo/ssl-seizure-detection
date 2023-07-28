# Libraries
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# Preprocessing functions for creating our pseudolabeled dataset to train our SSL model on
def create_samples(A, T, step_size):
    """ 
    Function that creates samples of window length T with step size step_size from a multivariate timeseries A.
    
    Args:
        A (numpy array): Multivariate timeseries of shape (N,M).
        T (int): Window length.
        step_size (int): Window step.
        
    Returns:
        samples (numpy array): Array of shape (K, N, T), where K is the number of samples.
        times (numpy array): Array of time indexes corresponding to each array in samples.
    
    """
    N, M = A.shape
    times = np.empty(0)
    samples = np.empty((0, N, T))
    
    # For each time index, take a sample of window length T, skip to next time index by the stepsize
    for i in range(0, M - T + 1, step_size):
        new_array = A[:, i:i+T]
        samples = np.concatenate((samples, new_array[np.newaxis, :, :]), axis = 0)
        times = np.hstack((times, i))
    
    return samples, times



def sample_pairs(samples, times, tau_pos, tau_neg):
    """ 
    Creates unique sample pairs from a list of samples and their corresponding time indexes.

    Args:
        samples (numpy array): Array of shape (K, N, T).
        times (numpy array): Array of time indexes corresponding to each array in samples.

    Returns:
        sample_pairs (numpy array): Array of unique sample pairs of shape (K*(K-1), N, T) numpy arrays with indices.
        time_differences (numpy array): Array of absolute value time index differences for each pair in sample_pairs.
    """

    K, N, T = samples.shape
    sample_pairs = np.empty((0, 2, N, T))
    time_differences = np.empty(0)
    
    for i in range(K):
        for j in range(K):
            diff = np.abs(times[i] - times[j])
            if (i != j) & ((diff <= tau_pos) | (diff > tau_neg)):
                x_i = samples[i]
                x_j = samples[j]
                pair = np.stack((x_i, x_j), axis = 0)
                sample_pairs = np.concatenate((sample_pairs, pair[np.newaxis, :, :]), axis = 0)
                time_differences = np.hstack((time_differences, diff))
        
    return sample_pairs, time_differences



def pseudolabels(sample_pairs, time_differences, tau_pos, tau_neg):
    """
    Function that pseudolabels unique sample pairs 0, 1, or none depending on their starting time indices.

    Args:
        sample_pairs (numpy array): Array of unique sample pairs of shape (K*(K-1), N, T) numpy arrays with indices.
        time_differences (numpy array): Array of absolute value time index differences for each pair in sample_pairs.
        tau_pos (_type_): Positive context threshold.
        tau_neg (_type_): Negative context threshold.
    """

    pseudolabels = np.empty(0)
    for i in range(len(sample_pairs)):
        if time_differences[i] <= tau_pos:
            pseudolabels = np.hstack((pseudolabels, 1))
        elif time_differences[i] > tau_neg:
            pseudolabels = np.hstack((pseudolabels, 0))

    return pseudolabels



def dataloader(A, T, step_size, tau_pos, tau_neg, batch_size, shuffle = True, testing = False):
    """
    Function that creates a data loader from a multivariate timeseries A.

    Args:
        A (numpy array): Multivariate timeseries of shape (N,M).
        T (int): Window length.
        step_size (int): Window step.
        tau_pos (int): Positive context threshold.
        tau_neg (int): Negative context threshold.
        batch_size (int): Batch size for data loader.

    Returns:
        data_loader (torch.utils.data.DataLoader): Data loader of samples and pseudolabels.
    """
    N, M = A.shape
    
    # Create samples, sample pairs, and pseudolabels
    samples, times = create_samples(A, T, step_size)
    pairs, time_differences = sample_pairs(samples, times, tau_pos, tau_neg)
    labels = pseudolabels(pairs, time_differences, tau_pos, tau_neg)
    
    # Convert to torch tensors
    X = torch.tensor(pairs, dtype=torch.float64)
    Y = torch.tensor(labels, dtype=torch.uint8)
    
    # Create dataset
    dataset = TensorDataset(X, Y)
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    if testing == True:
        return X, Y, dataset, data_loader
    else:
        return dataset, data_loader




def adjaceny_to_edge_index(A):
    """ 
    Converts the binary adjacency matrix to the edge index that pytorch_geometric uses to track edges.

    Args:
        A (torch.Tensor): Shape: (num_nodes, num_nodes), the binary adjaceny matrix.
    
    Returns:
        edge_index (torch.Tensor): Shape: (2, num_edges), the edge index matrix, where each column represents a directed edge.
    """
    edge_index = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    return edge_index


def in_graph_rep():
    # TODO: Implement input graph representation
    pass
    


# Test 1
A = torch.tensor([[0., 1., 1., 0.],
                  [1., 0., 0., 1.],
                  [1., 0., 0., 1.],
                  [0., 1., 1., 0.]])

print(adjaceny_to_edge_index(A))


# # Test 2
# # Number of electrodes
# N = 22
# # Number of discrete time points
# M = 101
# # Multivariate timeseries (iEEG data)
# A = np.random.randn(N, M)
# # Window length
# T = 6
# # Window step
# step_size = 10
# # Tau positive and tau negative
# tau_pos = 3
# tau_neg = 41
# batch_size = 32

# # Test

# A_x, A_y, dataset, data_loader = dataloader(A, T, step_size, tau_pos, tau_neg, batch_size, shuffle = False, testing = True)

# pair_1 = A_x[0][0]
# pair_2 = A_x[0][1]

# for inputs, labels in data_loader:
#     X_1, X_2 = inputs[:, 0], inputs[:, 1]
#     for i in range(len(X_1)):
#         print(X_1[i] == pair_1)
#         print(X_2[i] == pair_2)
#         break