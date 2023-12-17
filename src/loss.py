# Model
from torch_geometric.nn import NNConv, GATConv, global_mean_pool
from torch_geometric.graphgym.init import init_weights
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    # Assuming x is a square matrix
    n, m = x.shape
    assert n == m
    # Flatten the matrix, exclude the last element, and then reshape to have one extra column
    # This pushes diagonal elements to the last column, which are not selected in the next step
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegT1Loss(nn.Module):
    def __init__(self, loss_config = {"loss_coeffs":(25, 25, 1), "y_scale":True, "gamma":1, "epsilon":1e-4}):
        # loss_coeffs=(1, 1, 1), y_scale=True, gamma=1, epsilon=1e-4
        super(VICRegT1Loss, self).__init__()
        
        self.inv_coeff, self.var_coeff, self.covar_coeff = loss_config["loss_coeffs"]
        self.y_scale = loss_config["y_scale"]
        self.gamma = loss_config["gamma"]
        self.epsilon = loss_config["epsilon"]

    def forward(self, z1, z2, labels):

        # Number of features
        d = z1.shape[-1]
        
        # Temporal Invariance Loss
        sqr_diff = torch.norm(z1 - z2, p=2, dim=-1)**2

        if self.y_scale:
            inv_terms = labels * sqr_diff
        else:
            inv_terms = sqr_diff
        
        inv_loss = torch.mean(inv_terms)
        
        # Variance Loss
        # Compute the variance along the batch dimension (dim=0)
        var1 = z1.var(dim=0, unbiased=True)  # Set unbiased=False for population variance
        var2 = z2.var(dim=0, unbiased=True)

        # Compute S(z^j) for each feature dimension
        std1 = torch.sqrt(var1 + self.epsilon)
        std2 = torch.sqrt(var2 + self.epsilon)

        # Ensure each S(z^j) is at least gamma: the target standard deviation
        var_loss1 = torch.relu(self.gamma - std1).sum().div(d)
        var_loss2 = torch.relu(self.gamma - std2).sum().div(d)
        var_loss = (var_loss1 + var_loss2).div(2)
        
        # Covariance Loss
        # Compute the sample covariance matrices for z1 and z2
        z1_cov = torch.cov(z1.T, correction=1)
        z2_cov = torch.cov(z2.T, correction=1)
        
        # Sum the off diagonal elements
        covar_loss1 = (off_diagonal(z1_cov)**2).sum().div(d)
        covar_loss2 = (off_diagonal(z2_cov)**2).sum().div(d)
        covar_loss =  covar_loss1 + covar_loss2
        
        print(f"Invariance loss unscaled: {inv_loss:.4f}, Variance loss unscaled: {var_loss:.4f}, Covariance loss unscaled: {covar_loss:.4f}")
        print(f"Invariance loss: {self.inv_coeff*inv_loss:.4f}, Variance loss: {self.var_coeff*var_loss:.4f}, Covariance loss: {self.covar_coeff*covar_loss:.4f}")
        
        return self.inv_coeff * inv_loss + self.covar_coeff * covar_loss + self.var_coeff * var_loss