import pickle
import numpy as np
import tensorflow as tf
from dataloader_tf import dataloader_tf, dataset_tf, generator


# Iterate through the dataset
for item in dataset:
    graph1, graph2, label = item
    print('Graph 1:', graph1)
    print('Graph 2:', graph2)
    print('Label:', label.numpy())
    
