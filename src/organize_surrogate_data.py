import glob 
import os 
import numpy as np
import cv2
import random
from opts import OPT as opt

#Imagenet
path = '{opt.data_path}data/subset_Imagenet/class_0'
images = glob.glob(os.path.join(path,'*.JPEG'))
cnt =0
random.shuffle(images)
for img_file in images:

    img = cv2.imread(img_file)

    num=cnt//1000
        #create folders
    if not os.path.exists(f'{opt.data_path}data/subset_Imagenet_32/class_{num}'):
        os.makedirs(f'{opt.data_path}data/subset_Imagenet_32/class_{num}')
    if not os.path.exists(f'{opt.data_path}data/subset_Imagenet_64/class_{num}'):
        os.makedirs(f'{opt.data_path}data/subset_Imagenet_64/class_{num}')
    
    img32 = cv2.resize(img, (36,36))
    #get filename using os form img_file

    _,filename = os.path.split(img_file)
    #centercrop
    filename32= f'{opt.data_path}data/subset_Imagenet_32/class_{num}/{filename}'
    cv2.imwrite(filename32,img32[2:34,2:34,:])

    img64 = cv2.resize(img, (64,64))
    #centercrop
    filename64= f'{opt.data_path}data/subset_Imagenet_64/class_{num}/{filename}'
    cv2.imwrite(filename64,img64[4:68,4:68,:])
    cnt+=1

#check is ok for imagenet
print(glob.glob('{opt.data_path}data/subset_Imagenet_64/*'))

for folder in glob.glob('{opt.data_path}data/subset_Imagenet_64/*'):
    print(len(glob.glob(folder+'/*')))

#COCO
path = '{opt.data_path}data/COCO_subset/'
images = glob.glob(os.path.join(path,'*.jpg'))
cnt =0
random.shuffle(images)
for img_file in images:

    img = cv2.imread(img_file)

    num=cnt//1000
        #create folders
    if not os.path.exists(f'{opt.data_path}data/COCO_subset_32/class_{num}'):
        os.makedirs(f'{opt.data_path}data/COCO_subset_32/class_{num}')
    if not os.path.exists(f'{opt.data_path}data/COCO_subset_64/class_{num}'):
        os.makedirs(f'{opt.data_path}data/COCO_subset_64/class_{num}')
    
    img32 = cv2.resize(img, (36,36))
    #get filename using os form img_file

    _,filename = os.path.split(img_file)
    #centercrop
    filename32= f'{opt.data_path}data/COCO_subset_32/class_{num}/{filename}'
    cv2.imwrite(filename32,img32[2:34,2:34,:])

    img64 = cv2.resize(img, (64,64))
    #centercrop
    filename64= f'{opt.data_path}data/COCO_subset_64/class_{num}/{filename}'
    cv2.imwrite(filename64,img64[4:68,4:68,:])
    cnt+=1

#check is ok for imagenet
print(glob.glob('{opt.data_path}data/COCO_subset_64/*'))

for folder in glob.glob('{opt.data_path}data/COCO_subset_64/*'):
    print(len(glob.glob(folder+'/*')))

#rdn_img
path = '{opt.data_path}data/subset_rnd_img/train/class0/'
images = glob.glob(os.path.join(path,'*.jpg'))
cnt =0
random.shuffle(images)
for img_file in images:

    img = cv2.imread(img_file)

    num=cnt//1000
        #create folders
    if not os.path.exists(f'{opt.data_path}data/subset_rnd_img_32/class_{num}'):
        os.makedirs(f'{opt.data_path}data/subset_rnd_img_32/class_{num}')
    if not os.path.exists(f'{opt.data_path}data/subset_rnd_img_64/class_{num}'):
        os.makedirs(f'{opt.data_path}data/subset_rnd_img_64/class_{num}')
    
    img32 = cv2.resize(img, (36,36))
    #get filename using os form img_file

    _,filename = os.path.split(img_file)
    #centercrop
    filename32= f'{opt.data_path}data/subset_rnd_img_32/class_{num}/{filename}'
    cv2.imwrite(filename32,img32[2:34,2:34,:])

    img64 = cv2.resize(img, (64,64))
    #centercrop
    filename64= f'{opt.data_path}data/subset_rnd_img_64/class_{num}/{filename}'
    cv2.imwrite(filename64,img64[4:68,4:68,:])
    cnt+=1

#check is ok for imagenet
print(glob.glob('{opt.data_path}data/subset_rnd_img_32/*'))

for folder in glob.glob('{opt.data_path}data/subset_rnd_img_32/*'):
    print(len(glob.glob(folder+'/*')))