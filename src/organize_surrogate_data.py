import glob 
import os 
import numpy as np
import random
from opts import OPT as opt

#Imagenet
path = f'{opt.data_path}/subset_Imagenet/'
print(path)
images = glob.glob(os.path.join(path,'*.JPEG'))
cnt =0
random.shuffle(images)
for img_file in images:

    num=cnt//1000
        #create folders
    if not os.path.exists(f'{opt.data_path}/subset_Imagenet_split/class_{num}'):
        os.makedirs(f'{opt.data_path}/subset_Imagenet_split/class_{num}')
    os.system(f'cp {img_file} {opt.data_path}/subset_Imagenet_split/class_{num}')
        
    cnt+=1

#check is ok for imagenet
print(glob.glob(f'{opt.data_path}/subset_Imagenet_split/*'))

for folder in glob.glob(f'{opt.data_path}/subset_Imagenet_split/*'):
    print(len(glob.glob(folder+'/*')))

#COCO
path = f'{opt.data_path}/COCO_subset/'
print(path)
images = glob.glob(os.path.join(path,'*.jpg'))
cnt =0
random.shuffle(images)
for img_file in images:

    num=cnt//1000
        #create folders
    if not os.path.exists(f'{opt.data_path}/subset_COCO_split/class_{num}'):
        os.makedirs(f'{opt.data_path}/subset_COCO_split/class_{num}')
    os.system(f'cp {img_file} {opt.data_path}/subset_COCO_split/class_{num}')
        
    cnt+=1

#check is ok for imagenet
print(glob.glob(f'{opt.data_path}/subset_COCO_split/*'))

for folder in glob.glob(f'{opt.data_path}/subset_COCO_split/*'):
    print(len(glob.glob(folder+'/*')))

#rnd_img
path = f'{opt.data_path}/subset_rnd_img/train/class0/'
print(path)
images = glob.glob(os.path.join(path,'*.jpg'))
cnt =0
random.shuffle(images)
for img_file in images:

    num=cnt//1000
        #create folders
    if not os.path.exists(f'{opt.data_path}/subset_rnd_img_split/class_{num}'):
        os.makedirs(f'{opt.data_path}/subset_rnd_img_split/class_{num}')
    os.system(f'cp {img_file} {opt.data_path}/subset_rnd_img_split/class_{num}')
        
    cnt+=1

#check is ok for imagenet
print(glob.glob(f'{opt.data_path}/subset_rnd_img_split/*'))

for folder in glob.glob(f'{opt.data_path}/subset_rnd_img_split/*'):
    print(len(glob.glob(folder+'/*')))