import sys
sys.path.append(r'C:\Users\xmoot\Desktop\VSCode\ssl-seizure-detection\relative_positioning\pytorch\pyg')

import numpy as np
import torch
from pyg_preprocess import graph_triplets, pseudo_data, convert_to_TripletData
from torch_geometric.data import Data


def test_graph_triplets():
    # Create a synthetic dataset with 10 graph representations
    # Note that the graph representations won't be in this format, but it doesn't matter as it only needs to work for general lists with entries [a,b,c,d].
    data = [
        [["gr1", "gr1", "gr1"], np.random.randint(2)],
        [["gr2", "gr2", "gr2"], np.random.randint(2)],
        [["gr3", "gr3", "gr3"], np.random.randint(2)],
        [["gr4", "gr4", "gr4"], np.random.randint(2)],
        [["gr5", "gr5", "gr5"], np.random.randint(2)],
        [["gr6", "gr6", "gr6"], np.random.randint(2)]
        ]
    # Call the function with the synthetic dataset
    result = graph_triplets(data, tau_pos=2, tau_neg=5)

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
    
    
test_graph_triplets()
test_convert_to_TripletData()