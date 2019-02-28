from __future__ import division
import torch
from torchvision import transforms
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import skimage.transform


class ResizeImage(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, images):
        ret = []
        for image in images:
            dtype = image.dtype
            height, width, _ = image.shape
            # scale the same for both height and width for fixed aspect ratio resize
            scale = 1

            # bound the smallest dimension to the size
            image_min = min(height, width)
            scale = max(1, self.size / image_min)
        
            # next, bound the largest dimension to the size
            # this must be done after bounding to the size
            image_max = max(height, width)
            if round(image_max * scale) > self.size:
                scale = self.size / image_max

            if scale != 1:
                image = skimage.transform.resize(
                    image, (round(height * scale), round(width * scale)),
                    order=1,
                    mode='constant',
                    preserve_range=True)
            image_square = self.square_pad_image(image, self.size)
            toPIL = transforms.ToPILImage()
            ret.append(toPIL(image_square.astype(dtype)))
        return ret


    def square_pad_image(self, image, size):
        height, width, _ = image.shape
        if (size < height) or (size < width):
            raise ValueError('`size` must be >= to image height and image width')
        pad_height = (size - height) / 2
        pad_top = math.floor(pad_height)
        pad_bot = math.ceil(pad_height)
        pad_width = (size - width) / 2
        pad_left = math.floor(pad_width)
        pad_right = math.ceil(pad_width)
        return np.pad(
            image, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
            mode='constant')



class JointRandomHorizontalFlip(object):
    """Randomly horizontally flips the given list of PIL.Image with a probability of 0.5
    """

    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs


