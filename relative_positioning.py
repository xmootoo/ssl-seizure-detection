# Libraries
import numpy as np


# Functions
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



def sample_pairs(samples, times):
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
            if i != j:
                x_i = samples[i]
                x_j = samples[j]
                pair = np.stack((x_i, x_j), axis = 0)
                sample_pairs = np.concatenate((sample_pairs, pair[np.newaxis, :, :]), axis = 0)
                time_differences = np.hstack((time_differences, np.abs(times[i] - times[j])))
        
    return sample_pairs, time_differences



# Function that pseudolabels the sample pairs 0, 1, or none depending on their starting time indices
def pseudolabel(sample_pairs, time_differences, tau_pos, tau_neg):
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
        else:
            pseudolabels = np.hstack((pseudolabels, None))

    return pseudolabels






# Testing
# Number of electrodes
N = 20
# Number of discrete time points
M = 50
# Multivariate timeseries (iEEG data)
A = np.random.randn(N, M)
# Window length
T = 10
# Window step
step_size = 10
# Tau positive and tau negative
tau_pos = 200
tau_neg = 700


# Test
samples, times = create_samples(A, T, step_size)

sample_pairs, time_differences = sample_pairs(samples, times)

pseudolabels = pseudolabel(sample_pairs, time_differences, tau_pos, tau_neg)
