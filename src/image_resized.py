import os
import glob
import tarfile
from PIL import Image
from resizeimage import resizeimage
import cv2
import math
import skimage.transform
import numpy as np

def bound_image_dim(image, min_size=None, max_size=None):
    if (max_size is not None) and \
       (min_size is not None) and \
       (max_size < min_size):
        raise ValueError('`max_size` must be >= to `min_size`')
    dtype = image.dtype
    (height, width, *_) = image.shape
    # scale the same for both height and width for fixed aspect ratio resize
    scale = 1
    # bound the smallest dimension to the min_size
    if min_size is not None:
        image_min = min(height, width)
        scale = max(1, min_size / image_min)
    # next, bound the largest dimension to the max_size
    # this must be done after bounding to the min_size
    if max_size is not None:
        image_max = max(height, width)
        if round(image_max * scale) > max_size:
            scale = max_size / image_max
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(height * scale), round(width * scale)),
            order=1,
            mode='constant',
            preserve_range=True)
    image_square = square_pad_image(image, 256)
    return image_square.astype(dtype)


def square_pad_image(image, size):
    (height, width, *_) = image.shape
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

def main():

	src_path = glob.glob("/home/afarahani/Projects/project2/dataset/data/data/*")
	dest_path = '/home/afarahani/Projects/project2/dataset/data/cropdata/'
	for item in src_path:
	    long_path, directory = os.path.split(item)
	    
	    img_files = glob.glob(os.path.join(item+'/*.png'))
	    
	    path_fname = []
	    fname = []
	    for file in img_files:
	        head, tail = os.path.split(file)
	        # same_path=head.split('/')[:-1]
	        #fname.append(tail)
	        image = cv2.imread(file)
	        new_image = bound_image_dim(image, 256, 256)
	        path_to_save = os.path.join(dest_path, directory)
	        if not os.path.exists(path_to_save):
	            os.makedirs(path_to_save)
	        cv2.imwrite(path_to_save+'/'+tail, new_image)

    mask_path = glob.glob("/home/afarahani/Projects/project2/dataset/masks/*")
    path_to_save = '/home/afarahani/Projects/project2/dataset/data/cropmask/'

    for item in mask_path:
        head, tail = os.path.split(item)
        mask_img = cv2.imread(item)
        new_mask = bound_image_dim(mask_img, 256, 256)
        cv2.imwrite(path_to_save+tail, new_mask)

if __name__ == '__main__':
	main()