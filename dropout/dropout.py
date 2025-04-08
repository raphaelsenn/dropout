import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float=0.5, inplace: bool=False):
        super().__init__()
        self.p = torch.Tensor([p])
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:                   # at training time
            p = torch.ones_like(x) * self.p
            r = torch.bernoulli(input=p)
            if self.inplace:
                return x.mul(r)
            return r * x
        return x * self.p                   # at testing time