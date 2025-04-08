# dropout
Implementation of the dropout technique described in the paper: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

![image](/res/figure_1.png)

What is the benefit of this repository? Just to teach myself how dropout works.

## Notes

## Why do we need dropout?

* Deep neural networks with a large number of parameters are very expressive models. With limited training data, however, these models tend to fit to the noise of the data. As a result they perform well on the training set and poorly on testing data (overfitting).

* With unlimited computation, the best way to "regularize" a fixed-sized model is to average the predictions of of multiple neural nets (bagging). Combining several models is most helpful when the individual models are different (different architecture or training on different data). But training many different architetures is hard and exhausting. Moreover, large neural networks require large amounts of data. So the normal form of bagging is infeasible.

* **Key idea:** Dropout is a technique that addressed both of these issues (preventing overfitting and combining different models). The term "dropout" refers to (randomly) dropping out units (input + hidden, along with thier connections) from the neural network during training.
It can be inteerpreted as a way of reularizing a neural network by adding noise to its hidden units.

* **In practice:** Most Deep Learning libraries automatically scale up the output of each
neuron during training by 1/p, such that no changes are required during inference.

## How does dropout work?

* In the simplest case, each unit is retained with a fixed probability $p \in (0, 1)$ independent of other units. $p=0.5$ seems to be close to optimal for a wide range of networks and tasks. For input units, however, the optimal probability of retention i usually closer to 1 then to 0.5.

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

### MNIST

Used settings:
```python
epochs = 48
batch_size = 64
torch.manual_seed(seed=42)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.01, momentum=0.95)
```

Evaluated on the test set:

| Method | Unit Type | Architecture | Error (%) | Accuracy (%) |
| ------------------------------------- | --------- | ------------ | ---- |------ |
| Dropout NN (pytorch built in dropout) | ReLU | 3 layers, 1024 units | 1.38 | 98.62|
| Dropout NN (DIY dropout) | ReLU | 3 layers, 1024 units | 1.22 | 98.78 |

Error from the paper is $1.25$% with the same architecture.


### CIFAR-10

Used settings:
```python
epochs = 48
batch_size = 64
torch.manual_seed(seed=42)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.01, momentum=0.95)
```


Evaluated on the test set:

| Method | Error (%) |
| ------- | -------- |  
| Conv Net + max pooling + dropout fully connected layers | TODO |
| Conv Net + max pooling + dropout in all layers | TODO |

Conv Net + max pooling (no dropout)
```text
(train report)  error: 0.086580 acc: 0.913420
(test report)   error: 0.261700 acc: 0.738300
```


```bibtex

```
