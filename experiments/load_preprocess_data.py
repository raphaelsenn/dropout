import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_cifar10() -> None:
    """
    Uses tf to load cifar10 datasets and applies ZCA whitening. 
    """
    # load cifar10 dataset 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize 
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0
    
    # apply ZCA whitening
    datagen_train = ImageDataGenerator(zca_whitening=True)
    datagen_test = ImageDataGenerator(zca_whitening=True)

    datagen_train.fit(X_train)
    X_train_zca = []

    for x_batch, _ in datagen_train.flow(X_train, y_train, batch_size=128, shuffle=False):
        X_train_zca.append(x_batch)
        if len(X_train_zca) * 128 >= len(X_train):
            break
    
    datagen_test.fit(X_test)
    X_test_zca = []
    for x_batch, _ in datagen_test.flow(X_test, y_test, batch_size=128, shuffle=False):
        X_test_zca.append(x_batch)
        if len(X_test_zca) * 128 >= len(X_test):
            break
    
    X_train_zca = np.concatenate(X_train_zca, axis=0)[:len(X_train)]
    X_test_zca = np.concatenate(X_test_zca, axis=0)[:len(X_test)]

    # Convert to PyTorch format: (N, C, H, W)
    X_train = torch.from_numpy(X_train_zca).permute(0, 3, 1, 2).float()
    y_train = torch.from_numpy(y_train.squeeze()).long()
    
    # Convert to PyTorch format: (N, C, H, W)
    X_test = torch.from_numpy(X_test_zca).permute(0, 3, 1, 2).float()
    y_test = torch.from_numpy(y_test.squeeze()).long()

    torch.save((X_train, y_train), 'cifar10_zca_train.pt')
    torch.save((X_test, y_test), 'cifar10_zca_test.pt')


def load_cifar10_zca(batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    """
    Returns trainloader and testloader of cifar10 dataset as a tuple.

    Before loading cifar10_zca, execute: preprocess_cifar10!
    """
    X_train, y_train = torch.load('cifar10_zca_train.pt', weights_only=False)
    X_test, y_test = torch.load('cifar10_zca_test.pt', weights_only=False)
    
    print(X_train.shape, y_train.shape) 
    print(X_test.shape, y_test.shape) 
    dataset_test = TensorDataset(X_test, y_test)
    dataset_train = TensorDataset(X_train, y_train)


    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    return dataloader_train, dataloader_test


def load_cifar10(batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    cifar10_train = CIFAR10(
        root='cifar10/',
        train=True,
        download=True,
        transform=transform)
    
    cifar10_test = CIFAR10(
        root='cifar10/',
        train=False,
        download=True,
        transform=transform)
    
    dataloader_train = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True)
    
    return dataloader_train, dataloader_test



def load_mnist(batch_size: int=64) -> tuple[DataLoader, DataLoader]:
    # loading and transforming the mnist dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),                     # greyscale [0, 255] -> [0, 1]
        transforms.Lambda(lambda x: x.view(-1))])   # shape [1, 28, 28] -> [1, 784]

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
    
    dataloader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return dataloader_train, dataloader_test