# Libraries
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader




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
        graph_rep_pairs (numpy array): List of graph representation pairs [[A, NF, EF], [A', NF', EF'], Y], where Y corresponds to
        the pseudolabel of the pair.
    """

    n = len(graph_reps)

    graph_rep_pairs = []

    # Get rid of the real labels
    data = [graph_reps[i][0] for i in range(n)]

    # Create pairs and pseudolabels
    # TODO: Make sure we do not create symmetric pairs, we only need one of [[A, NF, EF], [A', NF', EF'], Y] or 
    # [[A', NF', EF'], [A, NF, EF],  Y] but not both as the they both use the same encoder (redundant)
    for i in range(n):
        for j in range(n):
            diff = np.abs(i-j)
            if (i != j):
                if diff <= tau_pos:
                    graph_rep_pairs.append([data[i], data[j], 1])
                elif diff > tau_neg:
                    graph_rep_pairs.append([data[i], data[j], 0])
        
    return graph_rep_pairs

    
# Test case
# Graph representations with labels
graph_reps = [
    [["adj1", "nf1", "ef1"], 1],
    [["adj2", "nf2", "ef2"], 0],
    [["adj3", "nf3", "ef3"], 1],
    [["adj4", "nf4", "ef4"], 0]
]

# Positive and negative context thresholds
tau_pos = 1
tau_neg = 2

# Call the function
pairs = graph_pairs(graph_reps, tau_pos, tau_neg)

# Print the pairs
for pair in pairs:
    print(pair)
