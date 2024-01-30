# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
from models.allcnn import AllCNN
from torch import nn

from opts import OPT as opt
import os 

mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'tinyImagenet': (0.485, 0.456, 0.406),
        'VGG':(0.547, 0.460, 0.404)
        }

std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'tinyImagenet': (0.229, 0.224, 0.225),
        'VGG':(0.323, 0.298, 0.263)
        }



transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
])

transform_train_tiny = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])

transform_test_tiny = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[opt.dataset], std=std[opt.dataset])
    ])
def trainer(removed=None):
    # Initialize the model
    if opt.model == 'resnet18':
        model= torchvision.models.resnet18(pretrained=True).to('cuda')
    elif opt.model=='resnet34':
        model= torchvision.models.resnet34(pretrained=True).to('cuda')
    elif opt.model=='resnet50':
        model = torchvision.models.resnet50(pretrained=True).to('cuda')
    elif opt.model=='AllCNN':
        model = AllCNN(n_channels=3, num_classes=opt.num_classes).to('cuda')
    
    if opt.dataset == 'cifar10':
        os.makedirs('./weights/chks_cifar10', exist_ok=True)
        # Load CIFAR-10 data
        trainset = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_test)
        model.fc = nn.Linear(512, opt.num_classes).to('cuda')

    elif opt.dataset == 'cifar100':
        os.makedirs('./weights/chks_cifar100', exist_ok=True)
        trainset = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_test)
        if 'resnet' in opt.model:    
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')

    elif opt.dataset == 'tinyImagenet':
        #dataloader
        os.makedirs('./weights/chks_tiny', exist_ok=True)
        trainset = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train',transform=transform_train_tiny)
        testset = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images',transform=transform_test_tiny)
        if 'resnet' in opt.model:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
            model.maxpool = nn.Identity()
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')
    #dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=opt.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=opt.num_workers)

    epochs=50
    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5)
    if opt.dataset == 'tinyImagenet':
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #learning rate decay
    else:
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


    # Train the network
    best_acc = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        correct, total = 0, 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = 100 * correct / total
        train_loss = running_loss / len(trainloader)
        train_scheduler.step()
        torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/best_checkpoint_{opt.model}.pth')

        if epoch % 5 == 0:        
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Val Acc: %.3f, Best Acc: %.3f' % (epoch, train_loss, train_acc, val_acc, best_acc))
    return best_acc


if __name__ == '__main__':
    trainer()