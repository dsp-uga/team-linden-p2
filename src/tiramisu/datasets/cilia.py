'''
This code is mofified from
https://github.com/bfortuner/pytorch_tiramisu/blob/master/datasets/camvid.py
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
    '''
    __init__ starts a class.
    __getitem__ builds iterator of pairs of input and target images.
    __len__ returns the length of the dataset
    '''
    def __init__(self, root, split='train', joint_transform=None):
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
            # transform the img and mask into PIL images (for cropping etc.)
            # toPIL = transforms.ToPILImage()
            # img, mask = toPIL(img), toPIL(mask)

            if self.joint_transform:
                img, mask = self.joint_transform([img, mask])
            
            mask = self.to_tensor_transform(mask).long()
                
        img = self.to_tensor_transform(img)
            
        if self.split != 'test':
            return img, mask[0, :, :]
        else:
            return img
    
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
            for num in ['00', '20', '50', '70', '90']:
                x_tmp = glob(self.root + self.split + '/data/' + hashcode + '/frame00' + num + '.png')
                x = np.array([imread(f, pilmode='I') for f in x_tmp])
                x_imgs.append(x.mean(axis=0))
                mask_tmp = glob(self.root + self.split + '/masks/' + hashcode + '.png')
                mask = np.array([imread(f, pilmode='I') for f in mask_tmp])
                masks_imgs.append(mask)

        # reshape the input
        for i in range(len(x_imgs)):
            x_imgs[i] = x_imgs[i].reshape(x_imgs[i].shape + (1,))
            x_imgs[i] = x_imgs[i].astype(np.uint8)

        # reshape the mask
        for i in range(len(masks_imgs)):
            masks_imgs[i] = masks_imgs[i].reshape(masks_imgs[i][0].shape + (1, ))
            masks_imgs[i] = masks_imgs[i].astype(np.int32)
        
        return x_imgs, masks_imgs


    def get_test_input(self):
        """
        generate the test input for our model, self.split == 'test'
        return test_imgs, path for test images
        """
        assert self.split == 'test'
        x_imgs = []
        hashcodes = os.listdir(self.root + self.split + '/data/')

        for hashcode in hashcodes:
            x_tmp = glob(self.root + self.split + '/data/' + hashcode + '/frame0000.png')
            x = np.array([imread(f, pilmode='I') for f in x_tmp])
            x_imgs.append(x.mean(axis=0))

        # reshape the input
        for i in range(len(x_imgs)):
            x_imgs[i] = x_imgs[i].reshape(x_imgs[i].shape + (1,))
            x_imgs[i] = x_imgs[i].astype(np.uint8)

        
        return x_imgs