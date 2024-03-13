# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.allcnn import AllCNN

from opts import OPT as opt
import os 

from dsets import get_dsets_remove_class, get_dsets


mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'tinyImagenet': (0.485, 0.456, 0.406),
        }

std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'tinyImagenet': (0.229, 0.224, 0.225),
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
def trainer(class_to_remove, seed):
    # Initialize the model
    if opt.model == 'resnet18':
        model= torchvision.models.resnet18(pretrained=True).to('cuda')
    elif opt.model=='resnet34':
        model= torchvision.models.resnet34(pretrained=True).to('cuda')
    elif opt.model=='resnet50':
        model = torchvision.models.resnet50(pretrained=True).to('cuda')
    elif opt.model=='AllCNN':
        model = AllCNN(n_channels=3, num_classes=opt.num_classes).to('cuda')

    if opt.mode == 'HR':
        if opt.dataset == "cifar10":
            num=5000
        elif opt.dataset == "cifar100":
            num=5000
        elif opt.dataset == "tinyImagenet":
            num=10000

        file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_{num}_{opt.dataset}_seed_{seed}.txt'
        _, test_loader, _, train_retain_loader = get_dsets(file_fgt=file_fgt)
    if opt.mode == 'CR':
        _, _, _, train_retain_loader, _, test_retain_loader = get_dsets_remove_class(class_to_remove)
        #use test_loader the one with forget classes removed
        test_loader = test_retain_loader

    if opt.dataset == 'cifar10':
        os.makedirs('./weights/chks_cifar10', exist_ok=True)
    elif opt.dataset == 'cifar100':
        os.makedirs('./weights/chks_cifar100', exist_ok=True)
        if 'resnet' in opt.model:    
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
            model.maxpool = nn.Identity()
    elif opt.dataset == 'tinyImagenet':
        os.makedirs('./weights/chks_tiny', exist_ok=True)
        #dataloader
        if 'resnet' in opt.model:
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, opt.num_classes)).to('cuda')



    epochs=30
    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5)
    if opt.dataset == 'tinyImagenet':
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #learning rate decay
    else:
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


    # Train the network
    best_acc = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        correct, total = 0, 0
        for i, data in enumerate(train_retain_loader, 0):
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
        train_loss = running_loss / len(train_retain_loader)
        train_scheduler.step()
        if opt.mode == 'CR':
            torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/best_checkpoint_without_{class_to_remove}.pth')
        elif opt.mode == 'HR':
            torch.save(model.state_dict(), f'weights/chks_{opt.dataset}/chks_{opt.dataset}_seed_{seed}.pth')

        if epoch % 5 == 0:        
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Val Acc: %.3f, Best Acc: %.3f' % (epoch, train_loss, train_acc, val_acc))
    return best_acc


if __name__ == '__main__':
    for i in opt.seed:
        if opt.mode == 'CR':
            for class_to_remove in opt.class_to_remove:
                best_acc = trainer(class_to_remove=class_to_remove,seed=i)
        else:
            best_acc = trainer(seed=i,class_to_remove=None)