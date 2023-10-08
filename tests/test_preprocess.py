import sys
import os
sys.path.append("../src")


import numpy as np
import torch
# OR to import specific functions:
from preprocess import graph_triplets, pseudo_data, convert_to_TripletData, create_tensordata, graph_triplets_sampled, adj_to_edge_attr, build_K_n
from torch_geometric.data import Data


def test_graph_triplets():
    # Create a synthetic dataset with 10 graph representations
    # Note that the graph representations won't be in this format, but it doesn't matter as it only needs to work for general lists with entries [a,b,c,d].
    data = [
        [["1"], np.random.randint(2)],
        [["2"], np.random.randint(2)],
        [["3"], np.random.randint(2)],
        [["4"], np.random.randint(2)],
        [["5"], np.random.randint(2)],
        [["6"], np.random.randint(2)]
        ]
    # Call the function with the synthetic dataset
    result = graph_triplets(data, tau_pos=3, tau_neg=5, sample_ratio=0.9)

    # Print the result to see what the function returns
    for triplet in result:
        print(f"Graph 1: {triplet[0]}")
        print(f"Graph 2: {triplet[1]}")
        print(f"Graph 3: {triplet[2]}")
        print(f"Pseudo Label: {triplet[3]}\n")




def test_pseudo_data():
    # Step 1: Create a sample data list
    data = [
        [["gr1", "gr1", "gr1"], np.random.randint(2)],
        [["gr2", "gr2", "gr2"], np.random.randint(2)],
        [["gr3", "gr3", "gr3"], np.random.randint(2)],
        [["gr4", "gr4", "gr4"], np.random.randint(2)],
        [["gr5", "gr5", "gr5"], np.random.randint(2)],
        [["gr6", "gr6", "gr6"], np.random.randint(2)]
        ]
    
    # Step 2: Test the function with various inputs
    # Test with default parameters
    try:
        pairs = pseudo_data(data, tau_pos = 2, tau_neg = 3, stats = True, save = False, patientid = "patient", logdir = None, model = "relative_positioning")
        assert isinstance(pairs, list), "Output is not a list"
        
        # Check the length of the output
        assert len(pairs) > 0, "Output list is empty"
        
        # Check the structure of the first element in the output
        assert len(pairs[0]) == 3, "Output elements do not have the correct structure"
        print(pairs)

    except Exception as e:
        print(f"Test with default parameters failed with error: {e}")
    
    
    # Test with a different model
    try:
        triplets = pseudo_data(data, tau_pos = 2, tau_neg = 4, stats = True, save = False, patientid = "patient", logdir = None, model = "temporal_shuffling")
        assert isinstance(triplets, list), "Output is not a list"
        
        # Check the length of the output
        assert len(triplets) > 0, "Output list is empty"
        
        # Check the structure of the first element in the output
        assert len(triplets[0]) == 4, "Output elements do not have the correct structure"
        print(triplets)
        
    except Exception as e:
        print(f"Test with temporal shuffling failed with error: {e}")

    
    print("Tests completed")


class TripletData(Data):
    """
    Creates the torch_geometric data object for a triplets of graphs.
    
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        if key == 'edge_index3':
            return self.x3.size(0)
        return super().__inc__(key, value, *args, **kwargs)



def test_convert_to_TripletData():
    # Step 1: Create a sample data list
    data_list = [
        [
            [torch.tensor([[0, 1], [1, 2]]), torch.rand((3, 3)), torch.rand((2, 3))],
            [torch.tensor([[1, 0], [2, 1]]), torch.rand((3, 3)), torch.rand((2, 3))],
            [torch.tensor([[2, 0], [1, 2]]), torch.rand((3, 3)), torch.rand((2, 3))],
            torch.tensor([1])
        ] 
        for _ in range(3)
    ]
    
    # Step 2: Call the function with the sample data
    output = convert_to_TripletData(data_list, save=False)
    
    
    if torch.equal(output[0].x1,data_list[0][0][1]) and torch.equal(output[2].edge_index3, data_list[2][2][0]) and torch.equal(output[0].y, data_list[0][3]):
        print(output[0].x1)
        print(data_list[0][0][1])
        print(data_list[2][2][0]) 
        print(output[2].edge_index3)
        print(output[0].y)
        print(data_list[0][3])
        print("Test passed")
    
    
def test_create_tensordata(mode = "binary"):

    num_nodes = 4
    num_edges = int(num_nodes * (num_nodes - 1) / 2)
    num_node_features = 3
    num_edge_features = 2

    # Create data_list with the 3 examples
    if mode == "multi":
        data_list = [[[np.random.rand(num_nodes, num_nodes), np.random.rand(num_nodes, num_node_features), np.random.rand(num_edges, num_edge_features)], np.random.randint(3)] for i in range(6)]

    if mode == "binary":
        data_list = [[[np.random.rand(num_nodes, num_nodes), np.random.rand(num_nodes, num_node_features), np.random.rand(num_edges, num_edge_features)], np.random.randint(2)] for i in range(6)]
    
    return create_tensordata(num_nodes, data_list, complete=True, save=False, logdir=None, mode="multi")


def test_adj_to_edge_attr():

    # Adjacency matrix A (same for all cases)
    A = np.array([[0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0]])

    # Edge index (same for all cases). Complete graph K_4.
    edge_index = build_K_n(4)

    # Test Case 1: No edge features
    case1_edge_attr = None

    # Test Case 2: Edge features in FCN format (shape = (num_nodes, num_nodes))
    case2_edge_attr = np.random.rand(4, 4, 2)

    # Test Case 3: Edge features in PyG format (shape = (num_edges, 1))
    case3_edge_attr = np.random.rand(12, 2)

    test_cases = {
        "Case 1": (A, edge_index, case1_edge_attr),
        "Case 2": (A, edge_index, case2_edge_attr),
        "Case 3": (A, edge_index, case3_edge_attr),
    }

    # Case 1: No edge features.
    print("Case 1: No edge features.")
    print(adj_to_edge_attr(A, edge_index))

    # Case 2: Edge features in FCN format shape = (num_nodes, num_nodes, num_edge_features).
    print("\nCase 2: Edge features in FCN format shape = (num_nodes, num_nodes, num_edge_features).")
    edge_attr_new = adj_to_edge_attr(A, edge_index, case2_edge_attr, "FCN")
    for i in range(4):
        for j in range(4):
            if i != j:
                print(f"A[{i}, {j}]: {A[i, j]}")
                print(f"Old edge_attr[{i}, {j}]: {case2_edge_attr[i, j]}")
    for k in range(6):
        print(f"New edge_attr[{k}]: {edge_attr_new[k]}")

    # Case 3: Edge features in PyG format shape = (num_edges, num_edge_features).
    print("\nCase 3: Edge features in PyG format shape = (num_edges, num_edge_features).")
    edge_attr_new = adj_to_edge_attr(A, edge_index, case3_edge_attr, "PyG")
    for k in range(12):
        print(f"New edge_attr[{k}]: {edge_attr_new[k]}")
        print(f"Old edge_attr[{k}]: {case3_edge_attr[k]}")