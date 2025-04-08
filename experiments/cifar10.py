import torch
import torch.nn as nn

from dropout.dropout import Dropout


class ConvNetDropoutTorch(nn.Module):
    """
    Convolutional neural network using torch.nn.Dropout(p) in fc-layers.
        -> Good generalization 
    """ 
    
    def __init__(self) -> None:
        super().__init__() 

        self.layer1 = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2)) 

        self.layer3 = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3**2, 2048),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU())

        self.out = nn.Linear(2048, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(start_dim=1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.out(out)
        return out


class ConvNetDropoutDIY(nn.Module):
    """
    Convolutional neural network using DIY implementation of Dropout(p) in fc-layers
        -> Good generalization 
    """ 
    def __init__(self) -> None:
        super().__init__() 

        self.layer1 = nn.Sequential(
            nn.Dropout(0.8), 
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            Dropout(0.5),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2)) 

        self.layer3 = nn.Sequential(
            Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential(
            Dropout(0.5),
            nn.Linear(256 * 3**2, 2048),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU())

        self.out = nn.Linear(2048, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(start_dim=1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.out(out)
        return out