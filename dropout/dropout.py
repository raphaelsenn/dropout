import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float=0.5, in_place: bool=True):
        super().__init__()
        self.p = torch.Tensor([p])
        self.in_place = in_place

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training: return x
        p = torch.ones_like(x) * self.p
        r = torch.bernoulli(input=p)
        if self.in_place:
            return x.mul(r) 
        return r * x