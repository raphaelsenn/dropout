import torch
import torch.nn as nn


# avoid devision by zero.
BASE_EPSILON = 1e-8


class MaxNorm(object):
    """
    MaxNorm weight constraint. 
    
    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.

    Args:
        max_value: the maximum norm value for the incoming weights. Default: 2
        axis: integer, dim along which to calculate weight norms.
    """ 

    def __init__(self, max_value: float=2, dim: int=0):
        self.max_value = max_value
        self.dim = dim

    def __call__(self, module: nn.Module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            norms = torch.norm(w, dim=self.dim, keepdim=True) 
            desired = torch.clamp(norms, 0, self.max_value)
            module.weight.data.mul_(desired / (BASE_EPSILON + norms))