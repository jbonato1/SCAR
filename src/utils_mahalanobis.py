
import numpy as np
from opts import OPT as opt
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


def get_centroids_covMat(features,lab,num_classes):
    distribs = []
    cov_matrix_inv=[]

    for i in range(num_classes):
        samples = tuckey_transf(features[lab==i].to(opt.device))
        distribs.append(samples.mean(0))
        cov = torch.cov(samples.T)
        cov_shrinked = cov_mat_shrinkage(cov_mat_shrinkage(cov))
        cov_shrinked = normalize_cov(cov_shrinked)
        cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))

    distribs=torch.stack(distribs)
    cov_matrix_inv=torch.stack(cov_matrix_inv)
    return distribs,cov_matrix_inv