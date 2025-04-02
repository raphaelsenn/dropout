import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from experiments.mnist import NeuralNet
from experiments.mnist import DropoutNN2


def evaluate(
        model: nn.Module,
        dataloader: DataLoader):
    model.eval() 
    correct = 0 
    with torch.no_grad():
        for inputs, targets in dataloader:
            preds = model(inputs)
            correct += torch.sum(targets == torch.argmax(preds, dim=1))
    acc = correct / len(dataloader.dataset)
    error = 1 - acc
    return acc, error


def train(
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int=10,
        lr: float=0.1,
        verbose: bool=False) -> None:
    
    model.train()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # for epoch in range(epochs):
    for epoch in range(epochs):
        total_loss = 0.0
        error = 0.0
        for X_batch, y_batch in dataloader:
            # reset gradients 
            optimizer.zero_grad() 
            
            # make predictions
            pred = model.forward(X_batch)
            
            # calculate cross-entropy
            loss = criterion(pred, y_batch)
            
            # error measures (+loss) 
            total_loss += loss.item()
            error += torch.sum(y_batch != torch.argmax(pred, dim=1))

            # backpropagation
            loss.backward()

            # update parameters
            optimizer.step()

        if verbose:
            total_loss = total_loss / len(dataloader.dataset) 
            total_error = error / len(dataloader.dataset)
            total_acc = 1 - total_error 
            print(f'epoch: {epoch}\tloss: {total_loss:.04f}\terror: {total_error:.04f}\tacc: {total_acc:.04f}')


def train_mnist(model: nn.Module):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.view(-1))]) 

    mnist_train = MNIST(
        root='mnist/',
        train=True,
        download=True,
        transform=transform)
    
    mnist_test = MNIST(
        root='mnist/',
        train=False,
        download=True,
        transform=transform)
    
    dataloader_train = DataLoader(mnist_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(mnist_test, batch_size=64, shuffle=True)
    
    # start training 
    train(
        model=model,
        dataloader=dataloader_train,
        epochs=30,
        lr=0.001,
        verbose=True)

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    # create neural networks
    net_mnist = DropoutNN2(n_in=28*28, n_hidden=1024, n_out=10)
    # net_mnist = NeuralNet(n_in=28*28, n_hidden=1024, n_out=10)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.view(-1))]) 

    mnist_train = MNIST(
        root='mnist/',
        train=True,
        download=True,
        transform=transform)
    
    mnist_test = MNIST(
        root='mnist/',
        train=False,
        download=True,
        transform=transform)
    
    dataloader_train = DataLoader(mnist_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(mnist_test, batch_size=64, shuffle=True)
    
    # start training 
    train(
        model=net_mnist,
        dataloader=dataloader_train,
        epochs=30,
        lr=0.001,
        verbose=True)

    acc_train, error_train = evaluate(net_mnist, dataloader_train)
    acc_test, error_test = evaluate(net_mnist, dataloader_test)
    print(f'error: {error_train:04f}\tacc: {acc_train:04f}')
    print(f'error: {error_test:04f}\tacc: {acc_test:04f}')

