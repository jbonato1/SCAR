# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
from torch import nn
from opts import OPT as opt
from models.ViT import ViT_16_mod

def split_retain_forget_idx(dataset, idx_file):
    idx = np.loadtxt(idx_file, dtype=int)
    forget_idx = idx
    retain_idx = np.arange(len(dataset.targets))
    retain_idx = np.delete(retain_idx, forget_idx)
    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)
    return forget_set, retain_set

def split_retain_forget(dataset, class_to_remove):

    # find forget indices
    forget_idx = np.where(np.array(dataset.targets) == class_to_remove)[0]
    
    forget_mask = np.zeros(len(dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return forget_set, retain_set


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224,antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224,antialias=True),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
])

def trainer(removed=None):
    # Load CIFAR-10 data
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainset = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_test)

    #val_forget_set, val_retain_set = split_retain_forget(testset, removed)
    #forget_set, retain_set = split_retain_forget(trainset, removed)

    #trainloader = torch.utils.data.DataLoader(retain_set, batch_size=256, shuffle=True, num_workers=8)
    #testloader = torch.utils.data.DataLoader(val_retain_set, batch_size=256, shuffle=False, num_workers=8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    epochs=2

    #Initialize the model, loss function, and optimizer
    net=torchvision.models.vit_b_16(pretrained=True).to(opt.device)
    #freeze backbone
    for name,param in net.named_parameters(): 
        if param.requires_grad:
            param.requires_grad=False
    
    #cifar10 change to 10
    net.heads = nn.Sequential(nn.Dropout(0.4), nn.Linear(768, 100)).to(opt.device)
    for name,param in net.heads.named_parameters():
        if param.requires_grad:
            param.requires_grad=True
    net = ViT_16_mod(n_classes=100).to(opt.device)
    #freeze backbone
    for name,param in net.encoder.named_parameters(): 
        if param.requires_grad:
            param.requires_grad=False

    for name,param in net.heads.named_parameters():
        if param.requires_grad:
            param.requires_grad=True

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print('ST pt: ',val_acc)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)


    optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=5e-5,eps=1e-8,betas=(0.9,0.999))
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Train the network
    best_acc = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        correct, total = 0, 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            optimizer.zero_grad()
            outputs = net.forward(inputs)
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

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), 'chks_cifar100/best_checkpoint_ViT.pth')

        print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Val Acc: %.3f, Best Acc: %.3f' % (epoch, train_loss, train_acc, val_acc, best_acc))
    return best_acc
if __name__ == '__main__':

    trainer()