import os
import argparse
from distutils.dir_util import copy_tree
import random

def main(args):
    """
    Simple function that looks at the arguments passed, checks to make sure
    everything expected exists, and then defines a validation data set for
    later processing by TrainTiramisu.py and TestTiramisu.py
    """

    args.rootDir = os.path.normpath(args.rootDir)
    args.outputDir = os.path.normpath(args.outputDir)

    # ensuring all expected files and directories exist
    if not os.path.exists(args.rootDir):
        raise Exception("ERROR: The dir '"+args.rootDir+"' doesn't exist")
    if not os.path.exists(args.rootDir+"/data"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/data' doesn't exist")
    if not os.path.exists(args.rootDir+"/masks"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/masks' doesn't " + \
                        "exist")
    if not os.path.exists(args.rootDir+"/test.txt"):
        raise Exception("ERROR: The file '"+args.rootDir+"/test.txt' " + \
                        "doesn't exist")
    if not os.path.exists(args.rootDir+"/train.txt"):
        raise Exception("ERROR: The dir '"+args.rootDir+"/train.txt' "+ \
                        "doesn't exist")
    
    # Make all output directories if needed
    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)
    if not os.path.exists(args.outputDir+"/test"):
        os.mkdir(args.outputDir+"/test")
    if not os.path.exists(args.outputDir+"/test/data"):
        os.mkdir(args.outputDir+"/test/data")
    if not os.path.exists(args.outputDir+"/validate"):
        os.mkdir(args.outputDir+"/validate")
    if not os.path.exists(args.outputDir+"/validate/data"):
        os.mkdir(args.outputDir+"/validate/data")
    if not os.path.exists(args.outputDir+"/validate/masks"):
        os.mkdir(args.outputDir+"/validate/masks")
    if not os.path.exists(args.outputDir+"/train"):
        os.mkdir(args.outputDir+"/train")
    if not os.path.exists(args.outputDir+"/train/data"):
        os.mkdir(args.outputDir+"/train/data")
    if not os.path.exists(args.outputDir+"/train/masks"):
        os.mkdir(args.outputDir+"/train/masks")
        
    # Read in test and train files
    testList = [line.rstrip('\n') for line in open(args.rootDir+"/test.txt")]
    trainList = [line.rstrip('\n') for line in open(args.rootDir+"/train.txt")]

    # Randomly suffle the train list
    random.seed(args.randSeed)
    random.shuffle(trainList)
    
    # Copy over all test data
    for name in testList:
        print("test: " + name)
        copy_tree(args.rootDir+"/data/"+name,args.outputDir+"/test/data/"+name)

    # Copy over validate data
    for name in trainList[:min(args.validNum,len(trainList))]:
        print("validate: " + name)
        copy_tree(args.rootDir+"/data/"+name,args.outputDir+ \
                  "/validate/data/"+name)
        os.copy(args.rootDir+"/masks/"+name+".png",args.outputDir+ \
                  "/validate/masks/"+name+".png")
    
    # Copy remaining data to train directory
    for name in trainList[args.validNum:]:
        print("train: " + name)
        copy_tree(args.rootDir+"/data/"+name,args.outputDir+ \
                  "/train/data/"+name)
        os.copy(args.rootDir+"/masks/"+name+".png",args.outputDir+ \
                  "/train/masks/"+name+".png")
        
    # Done!

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
    parser.add_argument('-v', '--validNum', required=True, type=int,
                        help='Size of validate set')
    parser.add_argument('-s', '--randSeed', required=True, type=int,
                        help='Random seed for defining validate set')
    parser.add_argument('-o', '--outputDir', required=True,
                        help='Root directory where new files and folders ' + \
                        'will be placed')
    
    args = parser.parse_args()
    main(args)
    
