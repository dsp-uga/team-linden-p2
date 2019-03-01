'''
This code is mofified from
https://github.com/bfortuner/pytorch_tiramisu/blob/master/datasets/camvid.py

We also got some inspiration from 
https://github.com/whusym/Cilia-segmentation-pytorch-tiramisu
'''

from glob import glob
import os
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils import data
from imageio import imread
from PIL import Image


class Cilia(data.Dataset):
    def __init__(self, root, split='train', joint_transform=None):
        """
        constructor
        @para: root, the root path for the data
        @para: split, either 'train', 'val', or 'test'
        @para: joint_transform, 
        """
        assert split in ('train', 'validate', 'test')
        self.root = root
        self.split = split
        self.joint_transform = joint_transform
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        if split != 'test':
            self.imgs, self.masks = self.get_train_input()
        else:
            self.imgs = self.get_test_input()
    
    def __getitem__(self, index):
        img = self.imgs[index]
        
        if self.split != 'test':
            mask = self.masks[index]
            if self.joint_transform:

                img, mask = self.joint_transform([transforms.ToPILImage()(img), transforms.ToPILImage()(mask)])
            
            mask = self.to_tensor_transform(mask).long()
                
        img = self.to_tensor_transform(img)
            
        return img, mask[0, :, :] if self.split != 'test' else img
    
    def __len__(self):
        return len(self.imgs)

    def get_train_input(self):
        """
        generate the train/validate input for our model, self.split == 'train' or 'validate'
        return type: tuple (x_imgs, mask_imgs), path for X train images and masks
        """
        assert self.split in ('train', 'validate')
        x_imgs, masks_imgs = [], []
        hashcodes = os.listdir(self.root + self.split + '/data/')

        # pick 5 frames from a video
        for hashcode in hashcodes:
            for num in range(5, 100, 10):
                x_tmp = glob(self.root + self.split + '/data/' + hashcode + '/frame00' + str(num).zfill(2) + '.png')
                x = np.array([imread(f, pilmode='I') for f in x_tmp])
                x = x.mean(axis=0)
                x = x.reshape(x.shape + (1, ))
                x_imgs.append(x.astype(np.uint8))

                mask_tmp = glob(self.root + self.split + '/masks/' + hashcode + '.png')
                mask = np.array([imread(f, pilmode='I') for f in mask_tmp])
                mask = mask.reshape(mask[0].shape + (1, ))
                masks_imgs.append(mask.astype(np.int32))
        
        return x_imgs, masks_imgs


    def get_test_input(self):
        """
        generate the test input for our model, self.split == 'test'
        return test_imgs, path for test images
        """
        assert self.split == 'test'
        x_imgs = []
        hashcodes = sorted(os.listdir(self.root + self.split + '/data/'))

        for hashcode in hashcodes:
            x_tmp = glob(self.root + self.split + '/data/' + hashcode + '/frame0000.png')
            x = np.array([imread(f, pilmode='I') for f in x_tmp])
            x = x.mean(axis=0)
            x = x.reshape(x.shape + (1, ))
            x_imgs.append(x.astype(np.uint8))
            
        return x_imgs