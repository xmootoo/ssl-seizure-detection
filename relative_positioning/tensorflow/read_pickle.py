import pickle
import numpy as np

# File path (PC)
path = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/pickle/jh101_grs.pickle"

# File path (Macbook)

# Load pickle file
f = pickle.load(open(path, "rb"))



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


# Convert FCNs to adjacency matrices
for i in range(len(f)):
    f[i][0][0] = adj(f[i][0][0], 0.3)
    
# Train