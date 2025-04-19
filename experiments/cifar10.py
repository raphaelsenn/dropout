import torch
import torch.nn as nn

from dropout.dropout import Dropout


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
