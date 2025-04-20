
import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transforms

import numpy as np



def CustomCIFAR10(Dataset):
    """CIFAR-10 dataset with ZCA whitenig"""

    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.images = torch.from_numpy(data['images']).float()
        self.labels = torch.from_numpy(data['labels']).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]


def zca_whitening(
        X: torch.Tensor,
        epsilon: float=1.0
        ) -> torch.Tensor:
    N = X.shape[0] 
    X = X.reshape(-1, 3 * 32 * 32)  # [N, 3, 32, 32] -> [N, 3072]

    X = X - X.mean(axis=0)

    cov = np.cov(X, rowvar=False)

    U, S, V = np.linalg.svd(cov)


    ZCA_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
    X_zca = X @ ZCA_matrix.T

    X_zca_tensor = torch.from_numpy(X_zca).view(N, 3, 32, 32).float()
    return X_zca_tensor

    # X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X.T).T
    # X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min()) 
    # return torch.from_numpy(X_ZCA).view((N, 3, 32, 32)).float()

# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_train = torchvision.datasets.CIFAR10(root='cifar10/', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='cifar10/', train=False, download=True, transform=transform)

X_train, X_test = cifar10_train.data, cifar10_test.data
y_train, y_test = cifar10_train.targets, cifar10_test.targets

X_train_zca = zca_whitening(X_train)
y_train = torch.tensor(y_train)

X_test_zca = zca_whitening(X_test)
y_test = torch.tensor(y_test)


torch.save((X_train_zca, y_train), 'cifar10_train_zca.pt')
torch.save((X_test_zca, y_test), 'cifar10_test_zca.pt')