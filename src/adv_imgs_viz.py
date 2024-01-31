import torch
import torchshow as ts
import torchvision

train_data = torchvision.datasets.ImageFolder(root='/home/jb/Documents/trick_distill/data_adv/cifar100_adv/')
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True,  num_workers=4)

for num,ex in enumerate(train_data_loader):
    if num==0:
        tensor = ex[0]
        ts.show(tensor)
        ts.save(tensor,'/home/jb/Documents/trick_distill/adv.png')