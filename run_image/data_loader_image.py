"""
Load image datasets
Supports: cifar10, cifar100, mnist, fashion_mnist, svhn, stl10
"""
from torchvision import datasets, transforms
import torch

def cifar10_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def fashion_mnist_data():
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    return train_dataset, test_dataset

def mnist_data():
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return train_dataset, test_dataset

def cifar100_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset



def svhn_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    
    return train_dataset, test_dataset


def stl10_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
    test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
    
    return train_dataset, test_dataset



def load_data(dataset_name):
    if dataset_name == 'cifar10':
        return cifar10_data()
    elif dataset_name == 'fashion_mnist':
        return fashion_mnist_data()
    elif dataset_name == 'mnist':   
        return mnist_data()
    elif dataset_name == 'cifar100':
        return cifar100_data()
    elif dataset_name == 'svhn':
        return svhn_data()
    elif dataset_name == 'stl10':
        return stl10_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def help_data():    
    print("Supported datasets:")
    print("  - svhn")
    print("  - cifar10")
    print("  - mnist")
    print("  - fashion_mnist")  
    print("  - cifar100")
    print("  - stl10")
    supported_datasets = ['svhn', 'cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'stl10']
    return supported_datasets

if __name__ == '__main__':
    supported_datasets = help_data()
    for dataset_name in supported_datasets:
        train_dataset, test_dataset = load_data(dataset_name)
        print(f"Dataset: {dataset_name}")
        print(len(train_dataset), len(test_dataset))
        print(train_dataset[0][0].shape)
        print(train_dataset[0][1])
        print("*****************************************")