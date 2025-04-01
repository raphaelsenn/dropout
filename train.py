import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from experiments.mnist import NeuralNet


def train(
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int=10,
        lr: float=0.1,
        verbose: bool=False) -> None:
    
    model.train()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    n_params = sum(params.numel() for params in model.parameters())
    n_weight_updates = 0

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
            total_error = error / len(dataloader_train.dataset)
            print(f'epoch: {epoch}\tloss: {total_loss}\terror: {total_error}')


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: x.view(-1))]) 

    mnist_train = MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transform)
    
    mnist_test = MNIST(
        root='data/',
        train=False,
        download=True,
        transform=transform) 

    dataloader_train = DataLoader(mnist_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(mnist_test, batch_size=64, shuffle=True)

    # create neural networks
    neuralnet = NeuralNet()
    train(
        model=neuralnet,
        dataloader=dataloader_train,
        epochs=10,
        lr=0.001,
        verbose=True) 
