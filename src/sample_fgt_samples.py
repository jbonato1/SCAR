import numpy as np
import torchvision
from opts import OPT
import os 

os.makedirs(f'{OPT.root_folder}/forget_id_files', exist_ok=True)

data_path = '~/data'
np.random.seed(42)

train_set_cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
train_set_cifar100 = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True)
train_set_tinyImagenet = torchvision.datasets.ImageFolder(root=data_path+'/tiny-imagenet-200/train/')

for num in [1000,5000]:
    forget_idx_cifar10 = np.random.choice(len(train_set_cifar10.targets), size=num, replace=False)
    forget_idx_cifar100 = np.random.choice(len(train_set_cifar100.targets), size=num, replace=False)
    forget_idx_tiny = np.random.choice(len(train_set_tinyImagenet.targets), size=num, replace=False)
    
    #save the indices into a txt file
    #np.savetxt(f'forget_idx_{num}_cifar10.txt', forget_idx_cifar10.astype(np.int64))
    #np.savetxt(f'forget_idx_{num}_cifar100.txt', forget_idx_cifar100.astype(np.int64))
    #np.savetxt(f'forget_idx_{num}_tinyImagenet.txt', forget_idx_tiny.astype(np.int64))


for i in [0,1,2,3,4,5,6,7,8,42]:
    data_path = '~/data'
    np.random.seed(i)

    train_set_cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
    train_set_cifar100 = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True)
    train_set_tinyImagenet = torchvision.datasets.ImageFolder(root=data_path+'/tiny-imagenet-200/train/')

    for num in [1000,5000]:
        forget_idx_cifar10 = np.random.choice(len(train_set_cifar10.targets), size=num, replace=False)
        forget_idx_cifar100 = np.random.choice(len(train_set_cifar100.targets), size=num, replace=False)
        forget_idx_tiny = np.random.choice(len(train_set_tinyImagenet.targets), size=2*num, replace=False)
        
        #save the indices into a txt file
        np.savetxt(f'{OPT.root_folder}/forget_id_files/forget_idx_{num}_cifar10_seed_{i}.txt', forget_idx_cifar10.astype(np.int64))
        np.savetxt(f'{OPT.root_folder}/forget_id_files/forget_idx_{num}_cifar100_seed_{i}.txt', forget_idx_cifar100.astype(np.int64))
        np.savetxt(f'{OPT.root_folder}/forget_id_files/forget_idx_{2*num}_tinyImagenet_seed_{i}.txt', forget_idx_tiny.astype(np.int64))
