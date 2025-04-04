import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float=0.5, in_place: bool=True):
        super().__init__()
        self.p = torch.Tensor([p])
        self.scaler = torch.Tensor([1/p])
        self.in_place = in_place

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:                   # at training time
            p = torch.ones_like(x) * self.p
            r = torch.bernoulli(input=p)
            if self.in_place:
                return x.mul(r) 
            return r * x
        return x * self.p                   # at testing time