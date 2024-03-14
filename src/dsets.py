import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Subset, random_split,Dataset
import torchvision
from torchvision import transforms
from opts import OPT as opt
import copy


def split_retain_forget(dataset, class_to_remove, trns_fgt=None):

    # find forget indices
    if type(class_to_remove) is list:
        forget_idx = None
        for class_rm in class_to_remove:
            if forget_idx is None:
                forget_idx = np.where(np.array(dataset.targets) == class_rm)[0]
            else:
                forget_idx = np.concatenate((forget_idx, np.where(np.array(dataset.targets) == class_rm)[0]))
            
    else:
        forget_idx = np.where(np.array(dataset.targets) == class_to_remove)[0]

    forget_mask = np.zeros(len(dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]
    dataset_clone=copy.deepcopy(dataset)
    if trns_fgt:
        dataset_clone.transform=trns_fgt
    forget_set = Subset(dataset_clone, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return forget_set, retain_set


def get_dsets_remove_class(class_to_remove):
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

    # download and pre-process CIFAR10
    transform_dset = transforms.Compose(
        [   transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )

    transform_test= transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )

    # we split held out - train
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_dset)
        test_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_test)
    elif opt.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_dset)
        test_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_test)
        
    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train',transform=transform_dset)
        test_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images',transform=transform_test)

    test_forget_set, test_retain_set = split_retain_forget(test_set, class_to_remove)
    forget_set, retain_set = split_retain_forget(train_set, class_to_remove, transform_test)

    # validation set and its subsets 
    all_test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_fgt_loader = DataLoader(test_forget_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_retain_loader = DataLoader(test_retain_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    # all train and its subsets
    all_train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    train_fgt_loader = DataLoader(forget_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)


    return all_train_loader,all_test_loader, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader




def get_dsets(file_fgt=None):
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

    transform_dset = transforms.Compose(
        [   transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )

    transform_test= transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset]),
        ]
    )
    
    if opt.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=opt.data_path, train=True, download=True, transform=transform_test)
        held_out = torchvision.datasets.CIFAR10(root=opt.data_path, train=False, download=True, transform=transform_test)
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_cifar10.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)

    elif opt.dataset=='cifar100':
        train_set = torchvision.datasets.CIFAR100(root=opt.data_path, train=True, download=True, transform=transform_test)
        held_out = torchvision.datasets.CIFAR100(root=opt.data_path, train=False, download=True, transform=transform_test)
        #use numpy modules to read txt file for cifar100
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_cifar100.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)

    elif opt.dataset == 'tinyImagenet':
        train_set = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/train/',transform=transform_test)
        held_out = torchvision.datasets.ImageFolder(root=opt.data_path+'/tiny-imagenet-200/val/images/',transform=transform_test)
        if file_fgt is None:
            forget_idx = np.loadtxt('./forget_idx_5000_tinyImagenet.txt').astype(np.int64)
        else:
            forget_idx = np.loadtxt(file_fgt).astype(np.int64)
    
    
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    
    ### get held out dataset for generating test and validation 
    
    test_set, val_set = random_split(held_out, [0.5, 0.5])
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, drop_last=True, shuffle=False, num_workers=opt.num_workers)

    

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]
    train_set_clone=copy.deepcopy(train_set)
    train_set_clone.transform=transform_test
    forget_set = Subset(train_set_clone, forget_idx)
    retain_set = Subset(train_set, retain_idx)


    train_forget_loader = DataLoader(forget_set, batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
    train_retain_loader = DataLoader(retain_set, batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)

    return train_loader, test_loader, train_forget_loader, train_retain_loader

    def __init__(self, load_synt=False, path=None, pretr_model=None, train_loader=None, save_folder=None,transform=None):
        self.transform=transform
        if load_synt:
            self.imgs=torch.load(os.path.join(path,"synt_imgs.pt"))
            self.targets=torch.load(os.path.join(path,"synt_labs.pt"))
        
    def __getitem__(self, index):
        img = self.imgs[index].to(opt.device)
        label = self.targets[index].to(opt.device)

        if self.transform:
            img = self.transform(img)

        return img, label
    def __len__(self):
        return len(self.imgs)
    
def get_surrogate(original_model=None):
    mean = {
            'subset_tiny': (0.485, 0.456, 0.406),
            'subset_Imagenet': (0.4914, 0.4822, 0.4465),
            'subset_rnd_img': (0.5969, 0.5444, 0.4877),
            'subset_COCO': (0.4717,0.4486,0.4089),
            'subset_gaussian_noise': (0,0,0)
            }

    std = {
            'subset_tiny': (0.229, 0.224, 0.225),
            'subset_Imagenet': (0.229, 0.224, 0.225),
            'subset_rnd_img': (0.3366, 0.3260, 0.3411),
            'subset_COCO': (0.2754, 0.2708, 0.2852),
            'subset_gaussian_noise': (1,1,1)
            }

    # download and pre-process CIFAR10
    transform_dset = transforms.Compose(
        [   
            transforms.Resize((64,64)) if opt.dataset == 'tinyImagenet' else transforms.Resize((32,32)),
            transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.surrogate_dataset],std[opt.surrogate_dataset]),
        ]
    )


    if opt.surrogate_dataset!="subset_gaussian_noise":
        set = torchvision.datasets.ImageFolder(root=os.path.join(opt.data_path,'surrogate_data/',opt.surrogate_dataset+'_split'), transform=transform_dset)
        if opt.surrogate_quantity == -1:
            subset = set
        else:
            class_list = [i for i in range(min(opt.surrogate_quantity,len(set.classes)))]
            idx = [i for i in range(len(set)) if set.imgs[i][1] in class_list]
            #build the appropriate subset
            subset = torch.utils.data.Subset(set, idx)
    else:
        #dataset from pt tensor
        subset = []
        if opt.surrogate_quantity == -1:
            opt.surrogate_quantity =10
       
        for i in range(opt.surrogate_quantity):
            fname = f"{opt.data_path}/surrogate_data/{opt.surrogate_dataset}_split/{i}/gaussian_noise_{i}.pt"
            print(fname)
            imgs = torch.load(fname)
            labels = torch.zeros(imgs.shape[0])
            subset.append(torch.utils.data.TensorDataset(imgs,labels))
        subset = torch.utils.data.ConcatDataset(subset)


    loader_surrogate = DataLoader(subset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    if opt.mode =='HR':
        bbone = torch.nn.Sequential(*(list(original_model.children())[:-1] + [torch.nn.Flatten()]))
        fc = original_model.fc
        bbone.eval()
        #forward pass into the original model 
        logits = []
        dset = []
        labels = []
        features_sur = []
        for img,lb in loader_surrogate:
            with torch.no_grad():
                output = original_model(img.to(opt.device))
                logits.append(output.detach().cpu())
                lb = torch.argmax(output,dim=1).detach().cpu()
                dset.append(img)
                labels.append(lb)
                features_sur.append(bbone(img.to(opt.device)).detach().cpu())

        logits = torch.cat(logits)
        dset = torch.cat(dset)
        labels = torch.cat(labels)
        features_sur=torch.cat(features_sur)

        clean_logits = []
        clean_labels = []
        clean_dset = []

        dataset_wlogits = custom_Dset_surrogate(dset,labels,logits)
        print('LEN surrogate',dataset_wlogits.__len__())
        
        class_sample_count = torch.zeros_like(labels)
        for i in range(opt.num_classes):
            class_sample_count[labels==i] = len(torch.where(labels==i)[0])
        #correct for undersampled output
        class_sample_count[class_sample_count<3]=5

        weights = 1 / torch.Tensor(class_sample_count)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,num_samples=dataset_wlogits.__len__(), replacement=True)
        loader_surrogate = DataLoader(dataset_wlogits, batch_size=opt.batch_size-512, num_workers=opt.num_workers,sampler=sampler)#
    return loader_surrogate


class custom_Dset_surrogate(Dataset):
    def __init__(self, dset,labels, logits,transf=None):
        self.dset = dset
        self.labels = labels
        self.logits = logits
        self.transf = transf


    def __len__(self):
        return self.dset.shape[0]

    def __getitem__(self, index):
        x = self.dset[index]
        y = self.labels[index]
        logit_x = self.logits[index]
        if self.transf:
            x=self.transf(x)
        return x, y,logit_x