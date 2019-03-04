from src.tiramisu.datasets import cilia, joint_transforms
from src.tiramisu.utils import training
from src.tiramisu.models import tiramisu
import adabound

import torch
from torchvision import transforms
from torch.utils import data
from imageio import imwrite, imread

from pathlib import Path
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import argparse

def main(args):

    args.rootDir = os.path.normpath(args.rootDir)
    args.outputDir = os.path.normpath(args.outputDir)
    
    # ensure the root directory has expected subdirectories
    if not os.path.exists(args.rootDir):
        raise Exception("ERROR: The dir '"+args.rootDir+"' doesn't exist")
    if not os.path.exists(args.rootDir+"/test/data"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/test/data' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/test/masks"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/test/masks' " + \
                        "doesn't exist")
    if not os.path.exists(args.targModel):
        raise Exception("ERROR: The target model file '"+args.targModel+"' "+ \
                        "doesn't exist") 
    if not os.path.exists(args.outputDir):
        raise Exception("ERROR: The result path '"+args.outputDir+"' does "+ \
                        "not exist")
    if not os.path.exists(args.rootDir+"/results"):
        os.mkdir(args.rootDir+"/results") 
    if not os.path.exists(args.rootDir+"/weights"):
        os.mkdir(args.rootDir+"/weights") 

    training.RESULTS_PATH = Path(args.rootDir+"/results/")
    training.WEIGHTS_PATH = Path(args.rootDir+"/weights/")
    
    test_cilia = cilia.Cilia(args.rootDir, 'test')
    test_loader = torch.utils.data.DataLoader(test_cilia, batch_size=1, \
                                              shuffle=False)
    
    ## Load the target model
    model = tiramisu.FCDenseNet103(n_classes=3, in_channels=1).cuda()
    model.load_state_dict(torch.load(args.targModel)['state_dict'])

    test_dir = sorted(os.listdir(args.rootDir + '/test/data/'))
    for i, img in enumerate(test_loader):
        pred = training.get_test_pred(model, img)
        pred_img = pred[0, :, :]
        imwrite(os.path.join(args.outputDir, test_dir[i] + '.png'), \
                pred_img.numpy().astype(np.uint8))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='This ' + \
            'is part of the UGA CSCI 8360 Project 2 - . Please visit our ' + \
            'GitHub project at https://github.com/dsp-uga/team-linden-p2 ' + \
            'for more information regarding data organization ' + \
            'expectations and examples on how to execute our scripts.')
    
    
    parser.add_argument('-r','--rootDir', required=True,
                        help='The base directory storing files and ' + \
                        'directories conforming with organization ' + \
                        'expectations, please visit out GitHub website')
    parser.add_argument('-tm', '--targModel', required=True,
                        help='A model file to define the CNN weights')
    parser.add_argument('-o', '--outputDir', required=True,
                        help='A model file to define the CNN weights')
    
    args = parser.parse_args()
    main(args)