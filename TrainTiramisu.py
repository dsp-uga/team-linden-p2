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
    
    # ensure the root directory has expected subdirectories
    if not os.path.exists(args.rootDir):
        raise Exception("ERROR: The dir '"+args.rootDir+"' doesn't exist")
    if not os.path.exists(args.rootDir+"/test/data"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/test/data' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/test/masks"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/test/masks' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/validate/data"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/validate/data' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/validate/masks"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/validate/masks' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/train/data"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/train/data' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/train/masks"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/train/masks' " + \
                        "doesn't  exist")
    if not os.path.exists(args.rootDir+"/results"):
        os.mkdir(args.rootDir+"/results") 
    if not os.path.exists(args.rootDir+"/weights"):
        os.mkdir(args.rootDir+"/weights") 

    training.RESULTS_PATH = Path(args.rootDir+"/results/")
    training.WEIGHTS_PATH = Path(args.rootDir+"/weights/")

    train_joint_transformer = transforms.Compose([
        joint_transforms.JointRandomSizedCrop(args.randCrop), 
        joint_transforms.JointRandomHorizontalFlip()
        ])
    train_cilia = cilia.Cilia(args.rootDir, joint_transform \
                              = train_joint_transformer)
    train_loader = data.DataLoader(train_cilia, batch_size = args.batchSize, \
                                   shuffle = True)
    
    val_cilia = cilia.Cilia(args.rootDir, 'validate')
    val_loader = torch.utils.data.DataLoader(val_cilia, \
                                             batch_size=args.batchSize, \
                                             shuffle=True)

    print("Train: %d" %len(train_loader.dataset.imgs))
    print("Val: %d" %len(val_loader.dataset.imgs))
    
    inputs, targets = next(iter(train_loader))
    print("Inputs: ", inputs.size())
    print("Targets: ", targets.size())
    
    figure, subplot = plt.subplots(1,2)
    subplot[0].imshow(inputs[0, 0, :, :], cmap = 'gray')
    subplot[1].imshow(targets[0, :, :], cmap = 'gray')
    
    # Define constants for later reference
    LR = args.lr
    LR_DECAY = args.lrDecay
    DECAY_EVERY_N_EPOCHS = args.decayOverEpochs
    N_EPOCHS = args.nEpochs    

    ## define the model
    model = tiramisu.FCDenseNet103(n_classes=3, in_channels=1).cuda()
    model.apply(training.weights_init)
    if args.adaBound:
        optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, \
                                      final_lr=args.finalLr)
    if args.torchAdam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, \
                                     weight_decay=args.weightDecay)

    criterion = torch.nn.NLLLoss().cuda()

    ## Now run through the epochs
    train_acc, val_acc = [], []
    
    for epoch in range(1, N_EPOCHS+1):
        since = time.time()
    
        ### Train ###
        trn_loss, trn_err = training.train(model, train_loader, optimizer, \
                                           criterion, epoch)
        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, \
              trn_loss, 1 - trn_err))    
        time_elapsed = time.time() - since  
        print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
              time_elapsed % 60))
    
        ### Validate ###
        val_loss, val_err = training.test(model, val_loader, criterion, epoch)    
        print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1 - val_err))
        time_elapsed = time.time() - since  
        print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, \
              time_elapsed % 60))
        
        train_acc.append(1 - trn_err)
        val_acc.append(1 - val_err)
        
        ### Checkpoint ###    
        training.save_weights(model, epoch, val_loss, val_err)
    
        ### Adjust Lr ###
        training.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, 
                                      DECAY_EVERY_N_EPOCHS)

    ## Write final summary to output
    output_accu = np.zeros((len(train_acc), 3))
    output_accu[:, 0] = np.arange(1, 201)
    output_accu[:, 1] = np.array(train_acc)
    output_accu[:, 2] = np.array(val_acc)
    np.savetxt(args.rootDir + "/" + 'adabound_accuracy.txt', output_accu)
        
    # Done!

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='This ' + \
            'is part of the UGA CSCI 8360 Project 2 - . Please visit our ' + \
            'GitHub project at https://github.com/dsp-uga/team-linden-p2 ' + \
            'for more information regarding data organization ' + \
            'expectations and examples on how to execute our scripts.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-ab', '--adaBound', action='store_true', 
                       help='Use the adaBound CNN')
    group.add_argument('-ta', '--torchAdam', action='store_true', 
                       help='Use the torch Adam CNN')
    
    
    parser.add_argument('-r','--rootDir', required=True,
                        help='The base directory storing files and ' + \
                        'directories conforming with organization ' + \
                        'expectations, please visit out GitHub website')
    parser.add_argument('-bs', '--batchSize', type=int, default=1,
                        help='Size of batch between CNN weight adjustment')
    parser.add_argument('-rc', '--randCrop', type=int, default=256,
                        help='Random sized crop value for torchvision')
    
    parser.add_argument('-lr', '--lr', type=float, default=1e-4,
                        help='LR value for CNN')
    parser.add_argument('-flr', '--finalLr', type=float, default=1e-4,
                        help='Final LR value for AdaBound CNN')
    parser.add_argument('-de', '--decayOverEpochs', type=int, default=1,
                        help='Number of epochs crossed for decay')
    parser.add_argument('-lrd', '--lrDecay', type=float, default=0.995,
                        help='LR Decay Rate')
    parser.add_argument('-ne', '--nEpochs', type=int, default=1000,
                        help='Number of epochs to run')
    parser.add_argument('-wd', '--weightDecay', type=float, default=1e-4,
                        help='Weight decay in tourch Adam CNN')
    
    args = parser.parse_args()
    main(args)

