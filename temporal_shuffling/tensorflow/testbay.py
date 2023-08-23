import tensorflow as tf
import numpy as np

# Sample data
data = [
    [[np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],
     [np.array([7, 8]), np.array([9, 10]), np.array([11, 12])], 0],
    [[np.array([13, 14]), np.array([15, 16]), np.array([17, 18])],
     [np.array([19, 20]), np.array([21, 22]), np.array([23, 24])], 1]
]

# Convert to suitable structure
inputs = [tf.constant([gr_1, gr_2]) for gr_1, gr_2, y in data]
labels = [y for _, _, y in data]

# Create a dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

# Iterate through the dataset
for item in dataset:
    input_data, label = item
    print('Input:', input_data.numpy())
    print('Label:', label.numpy())