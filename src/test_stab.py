import numpy as np
import torch
from torchattacks import PGD
import os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models.resnet import resnet18,ResNet18_Weights
import torch.nn as nn  
from dsets import *

weights_pretrained = torch.load('/home/jb/Documents/MachineUnlearning/src/weights/chks_cifar100/best_checkpoint_resnet18.pth')
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, 100)) 
model.load_state_dict(weights_pretrained)
#set cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#for class 0 
model.eval()
#cifar100 dataloader
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

# download and pre-process CIFAR10

transform_dset = transforms.Compose(
    [   #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean['cifar100'],std['cifar100']),
    ]
)
def accuracy(net, loader,device):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total

data_path = '/home/jb/data'
train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_dset)
test_forget_set, test_retain_set = split_retain_forget(train_set, [0])
dloader = torch.utils.data.DataLoader(test_forget_set,batch_size=1024,shuffle=True)
print(len(dloader))
model.eval()
print(accuracy(model,dloader,device))
print(accuracy(model,dloader,device))
print(accuracy(model,dloader,device))
print(accuracy(model,dloader,device))