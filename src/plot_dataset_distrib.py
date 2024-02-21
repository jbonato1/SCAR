from dsets import get_dsets
import matplotlib.pyplot as plt
import numpy as np
from opts import OPT as opt
from utils import get_trained_model
import torch 
import torch.nn as nn

def get_surrogate(surrogate_dataset):
    
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision
    import os
    # download and pre-process CIFAR10
    transform_dset = transforms.Compose([   
        transforms.Resize((224,224),antialias=True),
            transforms.ToTensor(),
        ]
    )

    train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transforms.ToTensor())
    set = torchvision.datasets.ImageFolder(root=os.path.join(opt.data_path,'surrogate_data',surrogate_dataset+'_split'),transform=transform_dset)


    loader_surrogate = DataLoader(set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    loader_train = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return loader_surrogate,loader_train


train_surrogate_loader,train_set = get_surrogate('subset_Imagenet')

# pixels distribution
img_list = []
for img,_ in train_set:
    img_list.append(img)

img_all = torch.cat(img_list)


img_listSur = []
for img,_ in train_surrogate_loader:
    img_listSur.append(img)
img_allSur = torch.cat(img_listSur)

train_surrogate_loader,_ = get_surrogate('subset_COCO')
img_listSur1 = []
for img,_ in train_surrogate_loader:
    img_listSur1.append(img)
img_allSur1 = torch.cat(img_listSur1)

train_surrogate_loader,_ = get_surrogate('subset_rnd_img')
img_listSur2= []
for img,_ in train_surrogate_loader:
    img_listSur2.append(img)
img_allSur2= torch.cat(img_listSur2)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

import scipy
print(scipy.stats.ks_2samp(img_all[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(), img_allSur[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy()))
print(scipy.stats.ks_2samp(img_all[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(), img_allSur1[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy()))
print(scipy.stats.ks_2samp(img_all[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(), img_allSur2[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy()))
fig, ax = plt.subplots(figsize=(3.5,1.2),ncols=3, nrows=1,sharey=True)
print(img_allSur.max(),img_allSur.min())
ax[0].hist(img_allSur[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(),bins=256,alpha=.3,color='blue',density=True,label='Imagenet subset')
ax[1].hist(img_allSur1[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(),bins=256,alpha=.3,color='red',density=True,label='COCO subset')
ax[2].hist(img_allSur2[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(),bins=256,alpha=.5,color='green',density=True,label='Random Images')
ax[0].hist(img_all[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(),bins=256,alpha=.3,color='k',density=True,label='CIFAR-100')
ax[1].hist(img_all[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(),bins=256,alpha=.3,color='k',density=True,label='CIFAR-100')
ax[2].hist(img_all[:,0,:,:].mean(-1).mean(-1).flatten().detach().cpu().numpy(),bins=256,alpha=.3,color='k',density=True,label='CIFAR-100')
#write text on the plots
ax[0].text(0, 4, 'KS-test p<0.001', fontsize=6)
ax[1].text(0, 4, 'KS-test p<0.001', fontsize=6)
ax[2].text(0, 4, 'KS-test p<0.001', fontsize=6)


for i in range(3):
    ax[i].set_xlabel('Pixel value',fontsize=8)
    # ax[i].set_title(f'Ch {i+1}')
    #ax[i].set_xlim(-3,3)
    ax[i].set_xticks([0,0.5,1])
    ax[i].set_yticks([0,1,2,3,4])
ax[0].set_ylabel('Prob. density',fontsize=8)
ax[0].legend(loc='upper left',frameon=False,fontsize=6)
ax[1].legend(loc='upper left',frameon=False,fontsize=6)
ax[2].legend(loc='upper left',frameon=False,fontsize=6)
fig.savefig(f'dataset_distributions.png')

fig.savefig(f'dataset_distributions.svg')

