import torch
import torch.nn as nn
from typing import Optional

class Flatten(nn.Module):
    """Flattens the input tensor to (batch_size, -1)."""
    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        if self.full:
            return x.view(-1)
        return x.view(x.size(0), -1)

def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn: Optional[nn.Module] = None):
    """Sequence of batchnorm, dropout and linear layers."""
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0.0: 
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: 
        layers.append(actn)
    return layers
