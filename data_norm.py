import torch
import torch.nn as nn
 
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.scale * x_norm

        
class MMNorm(nn.Module):
    def __init__(self, grid_min=-2.0, grid_max=2.0):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.eps = 1e-8

    def forward(self, x):
        x_norm = (x - x.min()) / (x.max() - x.min() + self.eps)
        x_norm = x_norm * (self.grid_max - self.grid_min) + self.grid_min
        return x_norm
   
    '''def forward(self, x):
        # Normalize each sample independently over the feature dimension
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + self.eps)
        return x_norm * (self.grid_max - self.grid_min) + self.grid_min'''