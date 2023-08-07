import pickle
from rp_dataloader_tf import adj, graph_pairs


# File path (PC)
path = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/pickle/jh101_grs.pickle"

# File path (Macbook)
# path =


# Load graph representations
data = pickle.load(open(path, "rb"))


# Convert FCNs to adjacency matrices
for i in range(len(data)):
    data[i][0][0] = adj(data[i][0][0], 0.3)

    
# Create graph pairs
tau_pos = 6 // 0.12
tau_neg = 20 // 0.12
pseudo_data = graph_pairs(data, tau_pos, tau_neg)



