import pickle
from preprocess import convert_to_tf_tensors, pseudo_data


# File path (PC) for subject jh101
path = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_gr/jh101_grs.pickle"

# File path (Macbook) for subject jh101
# path = "/Users/xaviermootoo/Documents/Data/SSL-seizure-detection/pickle/jh101_grs.pickle"

# Directory for dumping pseudolabeled data
logdir = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_pseudolabeled/relative_positioning/"


# Load graph representations list of [[A, NF, EF], Y]]
data = pickle.load(open(path, "rb"))

# Convert A, NF, EF, Y to TensorFlow tensors
# data = convert_to_tf_tensors(data)

# Create pseudolabeled graph pairs and write file for subject jh101.   
pdata = pseudo_data(data, tau_pos = 12 // 0.12, tau_neg = (7 * 60) // 0.12, mode = "weighted", stats = False, save = True, patientid = "jh101_12s_7min_np_2", logdir=logdir)








# # Test case
# path_pc = "C:/Users/xmoot/Desktop/Data/ssl-seizure-detection/patient_pseudolabeled/jh101_12s_7min.pkl"
# data = pickle.load(open(path_pc, "rb"))



# # Print within [[[A, NF, EF], [A', NF', EF']] Y] format
# print(len(data[0]) == 2)
# print(len(data[0][0]) == 2)
# print(len(data[0][0][0]) == 3)

# # [A, NF, EF]
# print(data[0][0][0])

# # [A', NF', EF']
# print(data[0][0][1])

# # [Y]
# print(data[0][1][0])
