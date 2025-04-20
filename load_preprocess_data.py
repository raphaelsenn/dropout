
import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_cifar10():
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


def load_cifar10():
    X_train, y_train = torch.load('cifar10_zca_train.pt', weights_only=False)
    X_test, y_test = torch.load('cifar10_zca_test.pt', weights_only=False)
    
    print(X_train.shape, y_train.shape) 
    print(X_test.shape, y_test.shape) 
    dataset_test = TensorDataset(X_test, y_test)
    dataset_train = TensorDataset(X_train, y_train)


    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)
    
    return dataloader_train, dataloader_test


# preprocess_cifar10()