import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.maxnorm import MaxNorm                         # max-norm for regularization

from experiments.neural_networks import DropoutMLPtorch         # uses pytorch's nn.Dropout method
from experiments.neural_networks import DropoutMLPdiy           # uses diy implementation of dropout

from experiments.neural_networks import ConvNetDropoutTorch     # uses pytorch's nn.Dropout method
from experiments.neural_networks import ConvNetDropoutDIY       # uses diy implementation of dropout

from experiments.preprocess_load_data import load_cifar10, load_mnist

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

DATASET = 'cifar10'             # mnist or cifar10
DROPOUT = 'torch'               # torch or diy
ROOT_DATA = DATASET + '/'

epochs = 100                    # number of iterations
learning_rate = 0.001           # learning rate
momentum = 0.95                 # momentum for SGD
lamb = 0.001                    # l2 penalty on the weights
c = 4                           # max-norm weight constraint (i did not use maxnorm on MNIST)
batch_size = 32                 # batch size
seed = 42                       # random seed
num_threads = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
verbose = True

torch.manual_seed(seed=seed)
torch.set_num_threads(num_threads)


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
        ) -> tuple[float, float]:
    model.eval() 
    
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device) 
            preds = model(inputs)
            correct += torch.sum(targets == torch.argmax(preds, dim=1))

    model.train()

    acc = correct / len(dataloader.dataset)
    error = 1 - acc
    return acc, error


def train(
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int,
        lr: float,
        momentum: float,
        device: torch.device,
        verbose: bool
        ) -> None:
    model.train()
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    norm = MaxNorm(max_value=c)

    for epoch in range(epochs):

        total_loss, correct = 0.0, 0.0
        start_time = time.monotonic()
        for X_batch, y_batch in dataloader:
            # move tensors to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # reset gradients 
            optimizer.zero_grad() 
            
            # make predictions
            pred = model.forward(X_batch)
            
            # calculate cross-entropy
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            
            # error measures
            correct += torch.sum(y_batch == torch.argmax(pred, dim=1))

            # apply L2 regularization
            l2_norm = sum(torch.sum(torch.pow(p, 2)) for p in model.parameters())
            loss += lamb * l2_norm

            # backpropagation
            loss.backward()

            # update parameters
            optimizer.step()

            # apply max norm constraint
            model.apply(norm)
        
        end_time = time.monotonic()
        if verbose:
            total_loss = total_loss / len(dataloader.dataset) 
            total_acc = correct / len(dataloader.dataset)
            total_error = 1 - total_acc 
            epoch_time = end_time - start_time
            print(f'epoch: {epoch}\ttime: {epoch_time:.02f}s\tloss: {total_loss:.04f}\terror: {total_error:.04f}\tacc: {total_acc:.04f}')


if __name__ == '__main__':

    if DATASET == 'mnist':  # training on minst
        if DROPOUT == 'torch': 
            model = DropoutMLPtorch(n_in=28*28, n_hidden=1024, n_out=10)
        else:  
            model = DropoutMLPdiy(n_in=28*28, n_hidden=1024, n_out=10)

        # loading mnist
        dataloader_train, dataloader_test = load_mnist(ROOT_DATA)

    else: # training on cifar10
        if DROPOUT == 'torch': 
            model = ConvNetDropoutTorch()
        else:  
            model = ConvNetDropoutDIY()

        # loading cifar10        
        dataloader_train, dataloader_test = load_cifar10(ROOT_DATA)

    model.to(device)
    print(f'Using device: {device}\nStart training on dataset: {DATASET}')

    # finally start training on mnist
    train(
        model=model,
        dataloader=dataloader_train,
        epochs=epochs,
        lr=learning_rate,
        momentum=momentum,
        device=device,
        verbose=verbose)
    
    # evaluating 
    acc_train, error_train = evaluate(model, dataloader_train, device=device)
    acc_test, error_test = evaluate(model, dataloader_test, device=device)
    
    # printing results
    print(f'(train report)\terror: {error_train:04f}\tacc: {acc_train:04f}')
    print(f'(test report)\t error: {error_test:04f}\tacc: {acc_test:04f}')