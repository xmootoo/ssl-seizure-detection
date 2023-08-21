from numpy import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader


import tensorflow as tf
from sklearn.model_selection import train_test_split

def dataloaders_tf(data, batch_size=32, val_size=0.1, test_size=0.1, seed=0):
    """
    Creates TensorFlow data loaders for training, validation, and test sets from a given data list.
    
    Args:
        data (list): A list of entries where each entry is of the form [[graph_1, graph_2], label].
                     graph_1 and graph_2 are lists representing graphs, each of the form [A, X, E].
                     A, X, E are TensorFlow tensors, and label is a tensor.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        val_size (float, optional): The proportion of the data to include in the validation set. Defaults to 0.1.
        test_size (float, optional): The proportion of the data to include in the test set. Defaults to 0.1.
        seed (int, optional): Random seed for data splitting. Defaults to 0.
    
    Returns:
        tuple: A tuple containing three TensorFlow data loaders for the training, validation, and test sets.
               Each data loader yields batches of data in the form of ([graph_1, graph_2], label) tuples.
    """
    # Separate the inputs and labels
    inputs = [([graph_1, graph_2], label) for [graph_1, graph_2], label in data]
    
    # Split into training and temporary sets
    train_data, temp_data = train_test_split(inputs, test_size=val_size + test_size, random_state=seed)

    # Further split the temporary set into validation and test sets
    val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size), random_state=seed)

    # Convert to TensorFlow datasets
    train_loader = tf.data.Dataset.from_generator(lambda: iter(train_data), output_signature=([tf.TensorSpec(shape=(None, None), dtype=tf.float32)]*4, tf.TensorSpec(shape=(), dtype=tf.float32))).batch(batch_size)
    val_loader = tf.data.Dataset.from_generator(lambda: iter(val_data), output_signature=([tf.TensorSpec(shape=(None, None), dtype=tf.float32)]*4, tf.TensorSpec(shape=(), dtype=tf.float32))).batch(batch_size)
    test_loader = tf.data.Dataset.from_generator(lambda: iter(test_data), output_signature=([tf.TensorSpec(shape=(None, None), dtype=tf.float32)]*4, tf.TensorSpec(shape=(), dtype=tf.float32))).batch(batch_size)

    return train_loader, val_loader, test_loader




def dataloaders_torch(data, batch_size=32, val_size=0.1, test_size=0.1, seed=0):
    """
        Splits the data into PyTorch dataloaders for training, validation, and testing.
    
    Args:
        data (list): List of [graph_1, graph_2, label] where each graph is of the form [A, X, E] for 
        adjacency matrix, node features, and edge features respectively.
        batch_size (int, optional): Batch size. Defaults to 32.
        val_size (float, optional): Validation set proportion. Defaults to 0.1.
        test_size (float, optional): Test set proportion. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 0.
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    
    # Train, test, val split with seed
    indices = [i for i in range(len(data) // 2)]
    random.seed(seed)
    random.shuffle(indices)
    split1 = int(len(data) // 2 * (1 - val_size - test_size))
    split2 = int(len(data) // 2 * (1 - test_size))

    # Split evenly for the 3 classes
    train_indices = indices[:split1] + [indices[i] + len(data) // 2 for i in range(0, split1)]
    val_indices = indices[split1:split2] + [indices[i] + len(data) // 2 for i in range(split1, split2)]
    test_indices = indices[split2:] + [indices[i] + len(data) // 2 for i in range(split2, len(data) // 2)]

    
    # use torch's dataloader
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader
