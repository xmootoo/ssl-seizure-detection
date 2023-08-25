from numpy import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader


import tensorflow as tf
from sklearn.model_selection import train_test_split









def dataloaders_torch(data, batch_size=32, val_size=0.1, test_size=0.1, seed=0, num_examples=50000):
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
    # Data
    random.shuffle(data)
    data = data[0:num_examples]
    print("Number of examples is:", len(data))
    
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


# 4600^2 = 21,160,000. approx 900,000 examples


# Test case
# Test if test_loader has 0 entries (empty)