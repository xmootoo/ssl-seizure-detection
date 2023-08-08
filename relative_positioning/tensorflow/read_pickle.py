import pickle
from rp_preprocess_tf import adj, graph_pairs


# File path (PC) for subject jh101
path = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/pickle/jh101_grs.pickle"

# File path (Macbook) for subject jh101
path = "/Users/xaviermootoo/Documents/Data/SSL-seizure-detection/pickle/jh101_grs.pickle"


# Load graph representations
data = pickle.load(open(path, "rb"))


# Convert FCNs to adjacency matrices
for i in range(len(data)):
    data[i][0][0] = adj(data[i][0][0], 0.3)

    
# Create list of  graph pairs, with entries of the form [[A, NF, EF], [A', NF', EF'], Y]
tau_pos = 6 // 0.12
tau_neg = 20 // 0.12
pseudo_data = graph_pairs(data, tau_pos, tau_neg)

# Save the pseudolabeled data as a pickle file
with open('pseudo_data_jh101.pkl', 'wb') as f:
    pickle.dump(pseudo_data, f)
