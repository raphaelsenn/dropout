import torch
import torch.nn as nn
import torch.nn.functional as F

from dropout.dropout import Dropout


class DropoutMLPtorch(nn.Module):
    """
    One of the deep neural networks described in the paper (+ dropout).

    Using dropout from PyTorch

    Unit type: ReLU
    Architecture: 3 layers, 1024 units each
    Error rate in the paper: 1.25%
    """ 
    def __init__(self, n_in: int, n_out: int, n_hidden: int=1024) -> None:
        super().__init__() 
        
        self.dropout_input = nn.Dropout(p=0.2) 

        self.layer1 = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.Dropout(p=0.5))

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Dropout(p=0.5)) 

        self.out = nn.Linear(n_hidden, n_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout_input(x) 
        out = self.layer1(out)  # [N, 1024]
        out = self.layer2(out)  # [N, 1024]
        out = self.out(out)     # [N, 10]
        return out



class DropoutMLPdiy(nn.Module):
    """
    One of the deep dropout neural networks described in the paper.

    Using my own dropout method.

    Unit type: ReLU
    Architecture: 3 layers, 1024 units each
    Error rate in the paper: 1.25%
    """ 
    def __init__(self, n_in: int, n_out: int, n_hidden: int=1024) -> None:
        super().__init__() 
        self.dropout_input = Dropout(p=0.8)

        self.fc1 = nn.Linear(n_in, n_hidden, bias=True)
        self.dropout1 = Dropout(p=0.5) 
        
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.dropout2 = Dropout(p=0.5) 
        
        self.fc3 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.dropout3 = Dropout(p=0.5) 

        self.out = nn.Linear(n_hidden, n_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout_input(x) 
        
        out = self.fc1(out)     # [N, 1024]
        out = F.relu(out) 
        out = self.dropout1(out) 
        
        out = self.fc2(out)     # [N, 1024]
        out = F.relu(out) 
        out = self.dropout2(out) 
        
        out = self.fc3(out)     # [N, 1024]
        out = F.relu(out) 
        out = self.dropout3(out) 
        
        out = self.out(out)     # [N, 10]
        return out