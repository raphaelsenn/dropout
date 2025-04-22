import torch
import torch.nn as nn


class Dropout(nn.Module):
    """
    During training, randomly zeroes some of the elements of the
    input tensor with probability 1 - p (retaining elements from the input tensor with probability p).

    The zeroed element are chosen indepedently for each forward pass, they are sampled
    from a bernoulli distribution (r ~ Bernoulli(p)).

    Dropout prevents overfitting and provides a way of approximately combining
    exponentially many different neural network architectures efficiently.

    Args:
        p: probability of an element to be retained. Default: 0.5
    """

    def __init__(self, p: float=0.5):
        super().__init__()
        self.register_buffer('p', torch.tensor([p])) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:                   # at training time
            p = torch.ones_like(x) * self.p
            r = torch.bernoulli(input=p)
            return r * x
        return x * self.p                   # at testing time