'''
Preprocessing helper for creating a PyTorch Dataset class Cilia.
Official Document can be found at:
http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset

Inspired from:
https://github.com/bfortuner/pytorch_tiramisu/blob/master/datasets/camvid.py
https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/datasets/cityscapes.py

'''

import json
from numpy import array, zeros
from glob import glob
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from imageio import imread
from PIL import Image

def load_input(base, split):
    '''
    Helper function to get the foler path. Under base (could be train, val, or test).
    Return the path for under the folders of training imgs and masks.
    '''
    input_imgs = []
    masks_imgs = []
    all_hash = sorted(os.listdir(base + split + '/data/'))
    if split != 'test':
        for imgHash in all_hash:
            inputs1 = glob(base + split + '/data/' + imgHash + '/frame0000.png')#input changed to input1
            inputs2 = glob(base + split + '/data/' + imgHash + '/frame0020.png')#added to increase the number of training set
            inputs3 = glob(base + split + '/data/' + imgHash + '/frame0050.png')#added
            inputs4 = glob(base + split + '/data/' + imgHash + '/frame0070.png')#added
            inputs5 = glob(base + split + '/data/' + imgHash + '/frame0090.png')#added
            input_imgs.append(array([imread(f, pilmode='I') for f in inputs1]).mean(axis=0))#input changed to input1
            input_imgs.append(array([imread(f, pilmode='I') for f in inputs2]).mean(axis=0))#added
            input_imgs.append(array([imread(f, pilmode='I') for f in inputs3]).mean(axis=0))#added
            input_imgs.append(array([imread(f, pilmode='I') for f in inputs4]).mean(axis=0))#added
            input_imgs.append(array([imread(f, pilmode='I') for f in inputs5]).mean(axis=0))#added

            masks1 = glob(base + split + '/masks/' + imgHash + '.png')
            masks2 = glob(base + split + '/masks/' + imgHash + '.png')
            masks3 = glob(base + split + '/masks/' + imgHash + '.png')
            masks4 = glob(base + split + '/masks/' + imgHash + '.png')
            masks5 = glob(base + split + '/masks/' + imgHash + '.png')
            masks_imgs.append(array([imread(f, pilmode='I') for f in masks1]))
            masks_imgs.append(array([imread(f, pilmode='I') for f in masks2]))
            masks_imgs.append(array([imread(f, pilmode='I') for f in masks3]))
            masks_imgs.append(array([imread(f, pilmode='I') for f in masks4]))
            masks_imgs.append(array([imread(f, pilmode='I') for f in masks5]))

    else:    
        for imgHash in all_hash:
                inputs = glob(base + split + '/data/' + imgHash + '/frame0000.png')#input changed to input1
                input_imgs.append(array([imread(f, pilmode='I') for f in inputs]).mean(axis=0))#input changed to input1
                masks1 = glob(base + split + '/masks/' + imgHash + '.png')
                masks_imgs.append(array([imread(f, pilmode='I') for f in masks1]))


    # check whether if they are good
    if split != 'test':
        if len(input_imgs) == 0 or len(masks_imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        if len(input_imgs) != len(masks_imgs):
            raise RuntimeError('Must be the same amount of the input and mask images!')
        # reshape the mask
        for i in range(len(masks_imgs)):
            masks_imgs[i] = masks_imgs[i].reshape(masks_imgs[i][0].shape + (1, ))
            masks_imgs[i] = masks_imgs[i].astype(np.int32)
            # masks_imgs[i][masks_imgs[i] == 1] = 0
            # masks_imgs[i][masks_imgs[i] == 2] = 1

    # reshape the input
    for i in range(len(input_imgs)):
        input_imgs[i] = input_imgs[i].reshape(input_imgs[i].shape + (1,))
        input_imgs[i] = input_imgs[i].astype(np.uint8)

    if split != 'test':
        return input_imgs, masks_imgs
    else:
        return input_imgs


class CiliaData(data.Dataset):
    '''
    From https://github.com/bfortuner/pytorch_tiramisu/blob/master/datasets/camvid.py and 
    https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/datasets/cityscapes.py
    PyTorch class of dataset for loading input and target data. 
    __init__ starts a class.
    __getitem__ builds iterator of pairs of input and target images.
    __len__ returns the length of the dataset
    '''
    def __init__(self, root, split='train', joint_transform=None,
                 input_transform=None, target_transform=None, remove_cell=False):
        self.root = root
        assert split in ('train', 'validate', 'test')
        self.split = split
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.remove_cell = remove_cell
        if split != 'test':
            self.imgs, self.masks = load_input(self.root, split)
        else:
            self.imgs = load_input(self.root, split)
    
    def __getitem__(self, index):
        
        img = self.imgs[index]
        
        if self.split != 'test':
            target = self.masks[index]
            if self.remove_cell:
                target [target == 1] = 0
            # transform the img and target into PIL images (for cropping etc.)
            toPIL = transforms.ToPILImage()
            img, target = toPIL(img), toPIL(target)
            # we need joint transform because we need to crop the same area
            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target)
            if self.target_transform is not None:
                target = self.target_transform(target).long()
                
        if self.input_transform is not None:
            img = self.input_transform(img)
            
        if self.split != 'test':
            return img, target[0, :, :]
        else:
            return img
    
    def __len__(self):
        return len(self.imgs)
