# Model
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
from torch_geometric.graphgym.init import init_weights
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegT1Loss(nn.Module):
    def __init__(self):
        super(VICRegT1Loss, self).__init__()
        pass
    
    
    def forward(z1, z2, batch):
        invar_loss = torch.mean(torch.norm(z1 - z2, dim=-1))
    