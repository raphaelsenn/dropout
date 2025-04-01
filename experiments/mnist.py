
import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Neural Network using pytorch's dropout technique
"""

class NeuralNet(nn.Module):
    """
    Neural Network using pytorch's dropout technique
    
    Unit type: ReLU
    Architecture: 3 layers, 1024 units
    Error rate in paper: 1.25%
    """ 
    
    def __init__(self) -> None:
        super().__init__() 
        
        self.fc1 = nn.Linear(784, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 1024, bias=True)
        self.fc3 = nn.Linear(1024, 1024, bias=True)
        self.fc4 = nn.Linear(1024, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)   # [N, 1024]
        out = self.fc2(out) # [N, 1024]
        out = self.fc3(out) # [N, 1024]
        out = self.fc4(out) # [N, 10]
        return out