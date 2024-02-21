from dsets import get_surrogate,get_dsets
import matplotlib.pyplot as plt
import numpy as np
from opts import OPT as opt
from utils import get_trained_model
import torch 
import torch.nn as nn
from torch.nn import functional as F


def cov_mat_shrinkage(cov_mat,gamma1=opt.gamma1,gamma2=opt.gamma2):
    I = torch.eye(cov_mat.shape[0]).to(opt.device)
    V1 = torch.mean(torch.diagonal(cov_mat))
    off_diag = cov_mat.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    V2 = (off_diag*mask).sum() / mask.sum()
    cov_mat_shrinked = cov_mat + gamma1*I*V1 + gamma2*(1-I)*V2
    return cov_mat_shrinked

def normalize_cov(cov_mat):
    sigma = torch.sqrt(torch.diagonal(cov_mat))  # standard deviations of the variables
    cov_mat = cov_mat/(torch.matmul(sigma.unsqueeze(1),sigma.unsqueeze(0)))
    return cov_mat


def mahalanobis_dist( samples, mean,S_inv):
    #check optimized version
    diff = F.normalize(tuckey_transf(samples), p=2, dim=-1)[:,None,:] - F.normalize(mean, p=2, dim=-1)
    right_term = torch.matmul(diff.permute(1,0,2), S_inv)
    mahalanobis = torch.diagonal(torch.matmul(right_term, diff.permute(1,2,0)),dim1=1,dim2=2)
    return mahalanobis


def tuckey_transf(vectors,beta=opt.beta):
    return torch.pow(vectors,beta)


original_model = get_trained_model()
original_model.to(opt.device)
original_model.eval()
dloader = get_surrogate(original_model=original_model,device=opt.device)

bbone = torch.nn.Sequential(*(list(original_model.children())[:-1] + [nn.Flatten()]))
fc = original_model.fc
bbone.eval()

# same for retain set
file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_5000_{opt.dataset}_seed_0.txt'
train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets(file_fgt=file_fgt)

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

distribs = []
cov_matrix_inv=[]

for i in range(100):
    samples = tuckey_transf(features_ret[lab_ret==i].to(opt.device))
    distribs.append(samples.mean(0))
    cov = torch.cov(samples.T)
    cov_shrinked = cov_mat_shrinkage(cov_mat_shrinkage(cov))
    cov_shrinked = normalize_cov(cov_shrinked)
    cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))

distribs=torch.stack(distribs)
cov_matrix_inv=torch.stack(cov_matrix_inv)

import matplotlib.pyplot as plt

# Create a figure and a grid of subplots
fig, axes = plt.subplots(4, 5, figsize=(15, 10),sharex=True,sharey=True)  # figsize is optional, for size of the whole grid


dict_maha = {}
for i in range(100):
    dists = mahalanobis_dist(features_ret[lab_ret==i].to(opt.device),distribs,cov_matrix_inv).T
    print(dists[:,i].mean(),dists[:,i].std())
    dict_maha[f'class_{i}']=[dists[:,i].mean().detach().cpu().numpy(),dists[:,i].std().detach().cpu().numpy()]

    if i <20:
        # Iterate over the grid
        ax = axes[i//5,i%5]
        ax.hist(dists[:,i].detach().cpu().numpy(),bins=30)
        ax.vlines(dists[:,i].mean().detach().cpu().numpy(),0,60,color='r')
        ax.vlines(dists[:,i].mean().detach().cpu().numpy()+2*dists[:,i].std().detach().cpu().numpy(),0,60,color='r')

plt.savefig('distances_distrib.png')

import pickle as pk 
with open('maha_dict_c100.pkl', 'wb') as file:
    # Use pickle to dump the dictionary to the file
    pk.dump(dict_maha, file)
