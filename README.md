# dropout
Implementation of the dropout technique described in the paper: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) *Srivastava et al., JMLR, 2014*

![Figure 1 from Srivastava et al., 2014](/res/figure_1.png)
Taken from *Srivastava et al., 2014*,  
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting",  
*Journal of Machine Learning Research*, 15(1):1929–1958, 2014.  


## Usage

```python
import torch
from dropout.dropout import Dropout


m = Dropout(p=0.8)  # unit is retained with probability 0.8
input = torch.randn(20, 16)
output = m(input)
```

## The Dropout Model

![Excerpt from Srivastava et al., 2014](/res/dropout_model.png)

![Excerpt from Srivastava et al., 2014](/res/dropout_model_2.png)
Taken from *Srivastava et al., 2014*,  
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting",  
*Journal of Machine Learning Research*, 15(1):1929–1958, 2014.  

**NOTE:** Dropout is only applied during training, but not druing the infernce/evaluation stage.

![[Figure 2 from Srivastava et al., 2014]](/res/figure_2.png)
Taken from *Srivastava et al., 2014*,  
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting",  
*Journal of Machine Learning Research*, 15(1):1929–1958, 2014.  

## Dropout vs. Inverted Dropout

* **NOTE** that dropout is only applied during training, but not druing the infernce/evaluation stage.

* Another way to achieve the same eﬀect is to scale up the retained activations by multiplying
by $1/p$ at training time and not modifying the weights at test time.



## PyTorch's Dropout vs. My Implementation

* Pytorch's implementation of dropout randomly zeroes some of the elements of the input tensor with probability $p$.

* My implementation of dropout randomly zeroes some of the elements of the input tensor with probability $1 - p$, so each unit is retained with a probability of $p$ (like in the original paper).

## Replicating Experiments from the Paper

I replicated some experiments from the paper and reused the same hyperparameters they used (roughly).  
I achieved results that were very similar to the original ones.

### MNIST

Used settings:

```python
epochs = 72 
batch_size = 64
torch.manual_seed(seed=42)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.01, momentum=0.95)
```

Evaluated on the test set:

| Method | Unit Type | Architecture | Error (%) | Accuracy (%) |
| ------------------------------------- | --------- | ------------ | ---- |------ |
| Dropout NN (pytorch built in dropout) | ReLU | 3 layers, 1024 units | 1.23 | 98.77 |
| Dropout NN (DIY dropout) | ReLU | 3 layers, 1024 units | 1.26 | 98.74 |

Error from the paper was $1.25$% with the same architecture and ~hyperparameters.


### CIFAR-10

Used settings:
```python
epochs = 48 
batch_size = 64
epsilon = 10e-7                 # ZCA whitening
lamb = 0.001                    # L2 regularization
p = (0.2, 0, 0, 0.5, 0.5, 0.5)  # dropout rates (retaining a unit)
torch.manual_seed(seed=42)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.001, momentum=0.95)
MaxNorm(max_value=4)            # max-norm weight constraint
```

Evaluated on the test set:

| Method | Error (%) |  Accuracy (%) |
| ------- | -------- | ------------- | 
| Conv Net + max pooling + dropout fully connected layers (torch) | 15.31 |  84.69 |
| Conv Net + max pooling + dropout fully connected layers (diy) | 15.91  |  84.09 |

Error from the paper was $14.32$% with the same architecture and ~hyperparameters.

### Info
In the deep convolutional network experiment from the paper, they applied a max-norm weight constraint and used ZCA whitening as a preprocessing step for CIFAR-10.

Since PyTorch doesn't natively support max-norm constraints or ZCA whitening, I implemented both from scratch.

Here is the max-norm constraint implementation:

```python
class MaxNorm(object):
    def __init__(self, max_value: float=2, dim: int=0):
        self.max_value = max_value
        self.dim = dim

    def __call__(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                norms = torch.norm(module.weight, dim=self.dim, keepdim=True) 
                desired = torch.clamp(norms, 0, self.max_value)
                module.weight.data.mul_(desired / (1e-8 + norms))
```

## Citations

```bibtex
@article{srivastava2014dropout,
    title   =   {{Dropout: A Simple Way to Prevent Neural Networks from Overfitting}},
    author  =   {Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov},
    journal = {Journal of Machine Learning Research},
    volume={15},
    pages={1929--1958},
    year    =   {2014}
}
```