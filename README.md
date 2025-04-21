# dropout
Implementation of the dropout technique described in the paper: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

![image](/res/figure_1.png)

What is the benefit of this repository? Just to teach myself how dropout works
(its pretty simple) :D

## Usage

```python
import torch
from dropout.dropout import Dropout


m = Dropout(p=0.8)  # unit is retained with probability 0.8
input = torch.randn(20, 16)
output = m(input)
```


## Notes

### Why do we need dropout?

* Deep neural networks with a large number of parameters are very expressive models. With limited training data, however, these models tend to fit to the noise of the data. As a result they perform well on the training set and poorly on the testing data (overfitting).

* With unlimited computation, the best way to "regularize" a fixed-sized model is to average the predictions of of multiple neural nets (bagging). Combining several models is most helpful when the individual models are different (different architecture and training on different data). But training many different architetures is hard and exhausting. Moreover, large neural networks require large amounts of data. So the "normal" form of bagging is infeasible.

* **Key idea:** Dropout is a technique that addressed both of these issues (preventing overfitting and combining different models). The term "dropout" refers to (randomly) dropping out units (input + hidden, along with thier connections) from the neural network during training.
It can be inteerpreted as a way of reularizing a neural network by adding noise to its hidden units.

* **In practice:** Most Deep Learning libraries automatically scale up the output of each
neuron during training by 1/p, such that no changes are required during inference.

### How does dropout work?

* In the simplest case, each unit is retained with a fixed probability $p \in (0, 1)$ independent of other units. $p=0.5$ seems to be close to optimal for a wide range of networks and tasks. For input units, however, the optimal probability of retention is usually closer to 1 then to 0.5.

* Applying dropout to a neural network amount to sampling a "thinned" network from it (Figure 1(b)).

* A neural net with $n$ units, can be seen as a collection of $2^n$ possible thinned neural networks. All these networks share weight, so the number of total parameters is still $O(n^2)$.

* Training a neural network with dropout can be seen as training a collection of $2^n$ thinned networks with extensive weight sharing.

* If a unit is retained with probability $p$, the outgoing weight of that unit are mutliplied by $p$ at test time.

* **NOTE** that dropout is only applied during training, but not druing the infernce/evaluation stage.

![image](/res/figure_2.png)

## The dropout model

![image](/res/dropout_model.png)

![image](/res/dropout_model_2.png)

## Dropout vs inverted dropout

* **NOTE** that dropout is only applied during training, but not druing the infernce/evaluation stage.

* Another way to achieve the same eﬀect is to scale up the retained activations by multiplying
by $1/p$ at training time and not modifying the weights at test time.


### Why scaling is important

* Using Dropout significantly affects the scale of the activations.

*  It is desired that the neurons throughout the model must receive the roughly same mean (or expected value) of activations during training and inference.

Assume this simplified neural network, with one hidden unit. Input unit_i = weight_i = 1 for all i = 0, ..., 99.

![image](/res/dropout_pen_and_paper.png)


## PyTorch vs my dropout implemention

* Pytorch's implementation of dropout randomly zeroes some of the elements of the input tensor with probability $p$.

* My implementation of dropout randomly zeroes some of the elements of the input tensor with probability $1 - p$, so each unit is retained with a probability of $p$ (like in the original paper).

## Replicating experiments from the paper
I replicated some experiments from the paper and reused the hyperparameters they used. I achived pretty much the same results like the original ones.

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
epsilon = 10e-7         # ZCA whitening
lamb = 0.001            # L2 regularization
torch.manual_seed(seed=42)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.001, momentum=0.95)
MaxNorm(max_value=4)    # max-norm weight constraint
```

Evaluated on the test set:

| Method | Error (%) |  Accuracy (%) |
| ------- | -------- | ------------- | 
| Conv Net + max pooling + dropout fully connected layers (torch) | 15.31 |  84.69 |
| Conv Net + max pooling + dropout fully connected layers (diy) | 15.91  |  84.09 |

Error from the paper was $14.32$% with the same architecture and ~hyperparameters.

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