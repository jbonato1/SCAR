import torch  
import os
# Set mean and standard deviation for each channel 
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.229, 0.224, 0.225]  
mean = [0,0,0]
std = [1,1,1]
# Generate a Gaussian noise tensor with the shape 10000x3x32x32 
# We will generate noise for each channel separately and then stack them 
noises = [torch.normal(mean=c_mean, std=c_std, size=(10000, 1, 32, 32)) for c_mean, c_std in zip(mean, std)]  
# Concatenate the noise tensors along the channel dimension to get the final noise tensor 
gaussian_noise = torch.cat(noises, dim=1)

#save in 10 folder with 1k images in each folder
for i in range(10):
    dname = f'/home/lsabetta/data/surrogate_data/subset_gaussian_noise_split/{i}/'
    os.makedirs(dname,exist_ok=True)
    torch.save(gaussian_noise[i*1000:(i+1)*1000],f'{dname}gaussian_noise_{i}.pt')