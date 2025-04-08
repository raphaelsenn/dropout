import torch
import torch.nn as nn
import torch.nn.functional as F

from dropout.dropout import Dropout


class DropoutMLPtorch(nn.Module):
    """
    Neural Network using pytorch's dropout technique
    
    Unit type: ReLU
    Architecture: 3 layers, 1024 units
    Error rate in paper: 1.25%
    """ 
    
    def __init__(self, n_in: int, n_out: int, n_hidden: int=1024) -> None:
        super().__init__() 
        self.dropout1 = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear(n_in, n_hidden, bias=True)

        self.dropout2 = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias=True)

        self.dropout3 = nn.Dropout(p=0.5) 
        self.fc3 = nn.Linear(n_hidden, n_hidden, bias=True)

        self.out = nn.Linear(n_hidden, n_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(x) 
        out = self.fc1(out)     # [N, 1024]
        out = F.relu(out) 

        out = self.dropout2(out) 
        out = self.fc2(out)     # [N, 1024]
        out = F.relu(out) 
        
        out = self.dropout3(out) 
        out = self.fc3(out)     # [N, 1024]
        out = F.relu(out) 
        
        out = self.out(out)     # [N, 10]
        return out



class DropoutMLPdiy(nn.Module):
    """
    Neural Network using pytorch's dropout technique
    
    Unit type: ReLU
    Architecture: 3 layers, 1024 units
    Error rate in paper: 1.25%
    """ 
    
    def __init__(self, n_in: int, n_out: int, n_hidden: int=1024) -> None:
        super().__init__() 
        self.dropout1 = Dropout(p=0.8) 
        self.fc1 = nn.Linear(n_in, n_hidden, bias=True)

        self.dropout2 = Dropout(p=0.5) 
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias=True)

        self.dropout3 = Dropout(p=0.5) 
        self.fc3 = nn.Linear(n_hidden, n_hidden, bias=True)

        self.out = nn.Linear(n_hidden, n_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(x) 
        out = self.fc1(out)     # [N, 1024]
        out = F.relu(out) 

        out = self.dropout2(out) 
        out = self.fc2(out)     # [N, 1024]
        out = F.relu(out) 
        
        out = self.dropout3(out) 
        out = self.fc3(out)     # [N, 1024]
        out = F.relu(out) 
        
        out = self.out(out)     # [N, 10]
        return out