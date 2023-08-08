import pickle
import pandas as pd

# Load pickle file of pseudolabeled data
# File path (PC) for subject jh101


# File path (Macbook) for subject jh101
path = "/Users/xaviermootoo/Documents/Data/SSL-seizure-detection/pickle/pseudo_data_jh101.pkl"


# Load graph representations
data = pickle.load(open(path, "rb"))

# Descriptive statistics
print("Number of examples: "  + str(len(data)))

# Count the number of occurrences of each value in 'y'
df = pd.DataFrame(data, columns=['col1', 'col2', 'y'])
counts = df['y'].value_counts()
print(counts)