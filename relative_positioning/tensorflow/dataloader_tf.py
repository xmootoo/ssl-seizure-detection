import tensorflow as tf
import numpy as np
import pickle

def generator(data):
    """Generates data for the dataset."""
    for (gr_1, gr_2, y) in data:
        yield (
            tuple(tf.convert_to_tensor(arr, dtype=tf.float32) for arr in gr_1),
            tuple(tf.convert_to_tensor(arr, dtype=tf.float32) for arr in gr_2),
            y
        )

def dataset_tf(data):
    """Creates a TensorFlow dataset from the data."""
    # Define the output types and shapes
    output_signature = (
        (tf.TensorSpec(shape=None, dtype=tf.float32),
         tf.TensorSpec(shape=None, dtype=tf.float32),
         tf.TensorSpec(shape=None, dtype=tf.float32)),
        (tf.TensorSpec(shape=None, dtype=tf.float32),
         tf.TensorSpec(shape=None, dtype=tf.float32),
         tf.TensorSpec(shape=None, dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    # Create a dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(data), # Using a lambda function to call generator with data
        output_signature=(output_signature[0], output_signature[1], output_signature[2])
    )

    return dataset


    

def dataloader_tf(dataset, val_size=0.2, test_size=0.1, batch_size=32, seed=16):
    # Calculate the total size of the dataset
    total_size = sum(1 for _ in dataset)

    # Calculate the sizes of the validation, test, and training sets
    val_size = int(total_size * val_size)
    test_size = int(total_size * test_size)
    train_size = total_size - val_size - test_size

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=total_size, seed=seed)

    # Create the training, validation, and test sets by taking the appropriate number of elements
    train_dataset = dataset.take(train_size).batch(batch_size)
    val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size).batch(batch_size)

    return train_dataset, val_dataset, test_dataset



# Testing real data
pseudodata_path = r"C:\Users\xmoot\Desktop\Data\ssl-seizure-detection\patient_pseudolabeled\relative_positioning\jh101_12s_7min_np_2.pkl"

# # Load data
data = pickle.load(open(pseudodata_path, "rb"))


dataset = dataset_tf(data, generator)
train_loader, val_loader, test_loader = dataloader_tf(dataset, batch_size=32, val_size=0.1, test_size=0.1, seed=16)



# Print examples to check the loaders
print("Start of train_loader:")
print(train_loader)

if not train_loader:
    print("train_loader is empty or None")

for i, batch in enumerate(train_loader):
    print(f"Batch {i}:")
    print(batch)

    if batch is None or len(batch) == 0:
        print("Empty or None batch")
        continue

    inputs, label = batch[:-1], batch[-1]

    if not inputs or not label:
        print("Missing inputs or label")
        continue

    for j in range(len(inputs[0])): # Adjust this line based on the actual structure of inputs
        print('Training Input:', [input[j] for input in inputs])
        print('Training Label:', label.numpy()[j])
        if j == 4:
            break




Tests
# Test data 
data = [
    [[np.array([[0.5645, 0.2412], [0.9563, 0.1425]]), np.array([0.8741, 0.3524, 0.4567]), np.array([[0.6324, 0.8632], [0.7841, 0.5124]])],
     [np.array([[0.7234, 0.4234], [0.7432, 0.9876]]), np.array([0.1234, 0.5310, 0.9865]), np.array([[0.6345, 0.2354], [0.7865, 0.5432]])], 0],
    [[np.array([[0.4523, 0.9876], [0.2413, 0.1532]]), np.array([0.3423, 0.6543, 0.7654]), np.array([[0.5432, 0.7654], [0.8675, 0.2134]])],
     [np.array([[0.1234, 0.9765], [0.7654, 0.2345]]), np.array([0.8765, 0.2345, 0.5432]), np.array([[0.1234, 0.4567], [0.7654, 0.9876]])], 0],
    [[np.array([[0.7654, 0.1234], [0.4567, 0.6543]]), np.array([0.9765, 0.1234, 0.8765]), np.array([[0.2345, 0.5678], [0.1234, 0.7654]])],
     [np.array([[0.5432, 0.7654], [0.8765, 0.4321]]), np.array([0.1234, 0.5678, 0.4321]), np.array([[0.7654, 0.1234], [0.2345, 0.6543]])], 0],
    [[np.array([[0.2345, 0.6789], [0.1234, 0.2345]]), np.array([0.8765, 0.1234, 0.4321]), np.array([[0.5432, 0.2345], [0.7654, 0.1234]])],
     [np.array([[0.9876, 0.1234], [0.4321, 0.5678]]), np.array([0.6543, 0.1234, 0.7654]), np.array([[0.8765, 0.2345], [0.1234, 0.4321]])], 0],
    [[np.array([0.5432, 0.8765, 0.1234]), np.array([0.7654, 0.2345]), np.array([0.4321, 0.5678, 0.8765])],
     [np.array([0.1234, 0.6789]), np.array([0.7654, 0.1234, 0.4567]), np.array([0.2345, 0.5678, 0.1234])], 1]
]

# Test for generator() and dataset_tf()
dataset = dataset_tf(data, generator)

# Iterate through the dataset
for item in dataset:
    graph1, graph2, label = item
    print('Graph 1:', graph1)
    print('Graph 2:', graph2)
    print('Label:', label.numpy())
    


# # Test for dataloader_tf()
# # Sample usage with the existing dataset, validation size 0.2, test size 0.2, batch size 1, seed 56
# train_loader, val_loader, test_loader = dataloader_tf(dataset, 0.2, 0.2, 1, seed=16)

# # Print examples to check the loaders
# for item in val_loader:
#     input_data, label = item[:-1], item[-1]
#     print('Training Input:', input_data)
#     print('Training Label:', label.numpy())



#<------------------------------------------------------------------------------------------->#
