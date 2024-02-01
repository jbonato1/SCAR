import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
# Define the dataset path
dataset_path = '/home/jb/data/fractaldb_cat60_ins1000/'

# Create a dataset instance
dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.ToTensor())

# Calculate the mean and standard deviation for normalization
mean = torch.stack([img.mean(1).mean(1) for img, _ in dataset]).mean(0)
std = torch.stack([img.std(1).std(1) for img, _ in dataset]).std(0)

print(mean)
print(std)