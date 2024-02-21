from dsets import get_surrogate,get_dsets
import matplotlib.pyplot as plt
import numpy as np
from opts import OPT as opt
from utils import get_trained_model
import torch 
import torch.nn as nn

from utils_mahalanobis import get_centroids_covMat,mahalanobis_dist
import torch
from torch.utils.data import DataLoader, Subset, random_split,Dataset
import torchvision
from torchvision import transforms
import os
def get_surrogate(retain_loader,original_model,device):
    
    bbone = torch.nn.Sequential(*(list(original_model.children())[:-1] + [nn.Flatten()]))
    fc = original_model.fc
    bbone.eval()
    # #collect info from retain
    # lab_ret = []
    # features_ret = []

    # for num, (img,lab) in enumerate(retain_loader):
    #     lab_ret.append(lab)
    #     with torch.no_grad():
    #         output = bbone(img.to(opt.device))
    #         features_ret.append(output.detach().cpu())

    # lab_ret = torch.cat(lab_ret)
    # features_ret = torch.cat(features_ret)

    # ####################### FILTERING ##########################

    # distribs,cov_matrix_inv = get_centroids_covMat(features_ret,lab_ret,num_classes=opt.num_classes)

    mean = {
            'subset_tiny': (0.485, 0.456, 0.406),
            'subset_Imagenet': (0.4914, 0.4822, 0.4465),
            'subset_rnd_img': (0.5969, 0.5444, 0.4877),
            'subset_COCO': (0.485,0.456,0.406)
            }

    std = {
            'subset_tiny': (0.229, 0.224, 0.225),
            'subset_Imagenet': (0.229, 0.224, 0.225),
            'subset_rnd_img': (0.3366, 0.3260, 0.3411),
            'subset_COCO': (0.229,0.224,0.225)
            }

    # download and pre-process CIFAR10
    transform_dset = transforms.Compose(
        
        [   transforms.Resize((64,64),antialias=True) if opt.dataset == 'tinyImagenet' else transforms.Resize((32,32),antialias=True),
            # transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.surrogate_dataset],std[opt.surrogate_dataset]),
        ]
    )
    transform_2 = transforms.Compose(
        
        [   
            transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )

    set = torchvision.datasets.ImageFolder(root=os.path.join(opt.data_path,'surrogate_data',opt.surrogate_dataset+'_split'),transform=transform_dset)
    if opt.surrogate_quantity == -1:
        subset = set
    else:
        class_list = [i for i in range(min(opt.surrogate_quantity,len(set.classes)))]
        idx = [i for i in range(len(set)) if set.imgs[i][1] in class_list]
        #build the appropriate subset
        subset = torch.utils.data.Subset(set, idx)

    loader_surrogate = DataLoader(subset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    #forward pass into the original model 
    logits = []
    dset = []
    labels = []
    features_sur = []
    for img,lb in loader_surrogate:
        with torch.no_grad():
            output = original_model(img.to(device))
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

    dataset_wlogits = custom_Dset_surrogate(dset,labels,logits)#,transf=transform_2)
    print('LEN surrogate',dataset_wlogits.__len__())
    
    class_sample_count = torch.zeros_like(labels)
    for i in range(opt.num_classes):
        class_sample_count[labels==i] = len(torch.where(labels==i)[0])
    #correct for undersampled output
    class_sample_count[class_sample_count<3]=5

    weights = 1 / torch.Tensor(class_sample_count)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,num_samples=dataset_wlogits.__len__(), replacement=True)
    loader_surrogate = DataLoader(dataset_wlogits, batch_size=opt.batch_size, num_workers=opt.num_workers,sampler=sampler)#

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
    
original_model = get_trained_model()
original_model.to(opt.device)
original_model.eval()
# same for retain set
file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_5000_{opt.dataset}_seed_0.txt'
train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets(file_fgt=file_fgt)

# same for unlearn
dloader = get_surrogate(train_retain_loader,original_model=original_model,device=opt.device)

bbone = torch.nn.Sequential(*(list(original_model.children())[:-1] + [nn.Flatten()]))
fc = original_model.fc
bbone.eval()

all_lab_distrib = []
features = []

for num, (img,lab, logits) in enumerate(dloader):
    print(num)
    all_lab_distrib.append(lab)

    with torch.no_grad():
        output = bbone(img.to(opt.device))
        features.append(output.detach().cpu())

all_lab_distrib = torch.cat(all_lab_distrib)
features = torch.cat(features)

A,B =torch.unique(all_lab_distrib,return_counts=True)
plt.plot(A.numpy(),B.numpy())
print(np.argsort(B.numpy()))
plt.savefig('counts.png')




lab_ret = []
features_ret = []

for num, (img,lab) in enumerate(train_retain_loader):
    print(num)
    lab_ret.append(lab)

    with torch.no_grad():
        output = bbone(img.to(opt.device))
        features_ret.append(output.detach().cpu())

lab_ret = torch.cat(lab_ret)
features_ret = torch.cat(features_ret)

####################### FILTERING ##########################

distribs,cov_matrix_inv = get_centroids_covMat(features_ret,lab_ret,num_classes=100)
#load dictionary
import pickle as pk
with open('maha_dict_c100.pkl', 'rb') as file:
    # Load the dictionary from the file using pickle
    maha_dict = pk.load(file)

clean_features = []
clean_all_lab_distrib = []
for i in range(100):
    feat_in = features[all_lab_distrib==i].to(opt.device)
    dists = mahalanobis_dist(feat_in,distribs,cov_matrix_inv).T
    buff_distance = dists[:,i]
    thresh = .2#maha_dict[f'class_{i}'][0]+2*maha_dict[f'class_{i}'][1]
    clean_features.append(feat_in[buff_distance<=thresh])
    clean_all_lab_distrib.append(i*torch.ones((clean_features[-1].shape[0],)))

###
####
#get tsne
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,random_state=42)

num_cl = 20
#filter first num_cl classes
lab_retF = lab_ret[lab_ret<num_cl]
features_retF = features_ret[lab_ret<num_cl]

all_lab_distribF = all_lab_distrib[all_lab_distrib<num_cl]
featuresF = features[all_lab_distrib<num_cl]


features_all = torch.cat([featuresF,features_retF])
N = featuresF.shape[0]

tsne_results = tsne.fit_transform(features_all)

res_retain = tsne_results[N:,:]
res_sur = tsne_results[:N,:]
#plot tsne_results
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light yellow-green
    '#9edae5'   # light blue-teal
]
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

fig, (ax1, ax2) = plt.subplots(figsize=(2.1, 1.9), ncols=1,nrows=2)
for i in range(num_cl):
    slot_i_ret = (lab_retF==i)
    slot_i_sur = (all_lab_distribF==i)
    ax1.scatter(res_retain[slot_i_ret,0],res_retain[slot_i_ret,1],c = colors[i],s=.5)
    ax2.scatter(res_sur[slot_i_sur,0],res_sur[slot_i_sur,1],c = colors[i],s=.5)

for a in [ax1,ax2]:
    a.set_xticks([])
    a.set_yticks([])
plt.savefig('tsne_inspection_orig.png')
plt.savefig('tsne_inspection_orig.svg')
###
    
# features = torch.cat(clean_features).detach().cpu()
# all_lab_distrib = torch.cat(clean_all_lab_distrib).detach().cpu()
# ####
# #get tsne
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2,random_state=42)

# num_cl = 20
# #filter first num_cl classes
# lab_retF = lab_ret[lab_ret<num_cl]
# features_retF = features_ret[lab_ret<num_cl]

# all_lab_distribF = all_lab_distrib[all_lab_distrib<num_cl]
# featuresF = features[all_lab_distrib<num_cl]


# features_all = torch.cat([featuresF,features_retF])
# N = featuresF.shape[0]

# tsne_results = tsne.fit_transform(features_all)

# res_retain = tsne_results[N:,:]
# res_sur = tsne_results[:N,:]
# #plot tsne_results
# fig, (ax1, ax2) = plt.subplots(1, 2)
# for i in range(num_cl):
#     slot_i_ret = (lab_retF==i)
#     slot_i_sur = (all_lab_distribF==i)
#     ax1.scatter(res_retain[slot_i_ret,0],res_retain[slot_i_ret,1],c = colors[i])
#     ax2.scatter(res_sur[slot_i_sur,0],res_sur[slot_i_sur,1],c = colors[i])
# plt.savefig('tsne_inspection.png')