import numpy as np
import torch
from torchattacks import PGD
import os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models.resnet import resnet18,ResNet18_Weights
import torch.nn as nn

def get_dset_stats(dloader,num_classes):
    """
    Compute the mean and standard deviation for each class and channel in the given dataloader.

    Args:
        dloader: The dataloader containing the inputs and targets.
        num_classes: The number of classes in the dataset.

    Returns:
        dict: A dictionary containing the mean and standard deviation for each class.
    """
    mean = []
    std = []
    label = []
    for inputs, targets in dloader:
        mean.append(inputs.mean(dim=[2,3]))
        label.append(targets)

    mean = torch.cat(mean,dim=0)
    label = torch.cat(label,dim=0)
    dict_stats={}
    for i in range(num_classes):
        mean_class = mean[label==i].mean(dim=0)
        std_class = mean[label==i].std(dim=0)
        dict_stats[i] = [mean_class,std_class]

    return dict_stats


def gen_adv_dataset(model,dloader,device,save_folder,samples_per_class=500,num_classes=100,verbose=False):

    dict_stats = get_dset_stats(dloader,num_classes=num_classes)
    images=[]
    labs=[]
    for i in range(num_classes):
        vec_rdn = torch.zeros((samples_per_class,3,32,32))

        for j in range(3):
            vec_rdn[:,j,:,:] = torch.randn((samples_per_class,32,32))*dict_stats[i][1][j]+dict_stats[i][0][j]

        labels = i*torch.ones((samples_per_class,),dtype=torch.int64).to(device)
        vec_rdn= vec_rdn.to(device)
        #torchattacks
        steps=50
        acc=0
        while acc<.8:
            atk = PGD(model, eps=190/255, alpha=2/225, steps=steps, random_start=True)
            atk.set_mode_targeted_by_label()
            adv_images = atk(vec_rdn, labels)
            
            adv_pred = model(adv_images)
            acc = torch.sum(torch.argmax(adv_pred,dim=1)==labels)/samples_per_class
            if verbose: print(f'class {i} step {steps} acc: {acc:.3f}') #print(acc)
            steps+=30
        #save tensor
        adv_images = adv_images.detach().cpu()
        images.append(adv_images)
        labs+=[i for _ in range(samples_per_class)]
        #gen folder name class_i in save_folder
    images=torch.cat(images,dim=0)
    labs=torch.tensor(labs)
    if save_folder is not None:
        torch.save(images,os.path.join(save_folder,'synt_imgs.pt'))
        torch.save(labs,os.path.join(save_folder,'synt_labs.pt'))
    return images, labs
        

#test function gen_adv_dataset
if __name__ == "__main__":

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


    data_path = '/home/jb/data'
    train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_dset)
    dloader = torch.utils.data.DataLoader(train_set,batch_size=1024)

    gen_adv_dataset(model,dloader,device,save_folder='/home/jb/Documents/trick_distill/data_adv/cifar100_adv/',samples_per_class=200,num_classes=100,verbose=True)