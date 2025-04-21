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


class ConvNetDropoutTorch(nn.Module):
    """
    One of the deep convolutional neural networks described in the paper (+ dropout).
    
    Using dropout from PyTorch

    Error from the paper on CIFAR-10: 12.61% 
    """ 
    
    def __init__(self) -> None:
        super().__init__() 

        self.dropout_input = nn.Dropout(0.1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.25))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5))

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3**2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.out = nn.Linear(2048, 10)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout_input(x) 
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.out(out)
        return out
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # dense unifrom, sigma=0.01
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)
            
            elif isinstance(m, nn.Linear):
                # dense unifrom sqrt fan_in, sigma=1.0
                bound = 1.0 / ((m.in_features) ** 0.5)
                nn.init.uniform_(m.weight, a=-bound, b=bound)

                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)


class ConvNetDropoutDIY(nn.Module):
    """
    One of the deep convolutional neural networks described in the paper (+ dropout).

    Using my own dropout method.

    Error from the paper on CIFAR-10: 12.61%
    """ 
 
    def __init__(self) -> None:
        super().__init__() 

        self.dropout_input = Dropout(0.9)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Dropout(0.75))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            Dropout(0.75))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            Dropout(0.5))

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3**2, 2048),
            nn.ReLU(),
            Dropout(0.5))
        
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            Dropout(0.5))

        self.out = nn.Linear(2048, 10)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout_input(x) 
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.out(out)
        return out
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # dense unifrom, sigma=0.01
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)
            
            elif isinstance(m, nn.Linear):
                # dense unifrom sqrt fan_in, sigma=1.0
                bound = 1.0 / ((m.in_features) ** 0.5)
                nn.init.uniform_(m.weight, a=-bound, b=bound)

                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)