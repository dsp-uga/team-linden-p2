"""
Code is referenced to https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
"""

import cv2
import os
import numpy as np
from glob import glob
from PIL import Image
import argparse


def getIOU(trainData, threshold):
    """
    calculate the IOU give the data and the propopsed threshold

    trainData: type: list, dimension: (N, 2), where the 1st column is scaled 
                     variance (0-256), the 2nd column is the label (0, 1, 2)
    threshold: current threshold
    return, type, float, the IOU
    """
    intersection, union = 0, 0
    for fea, label in trainData:
        if label == 2 or fea >= threshold:
            union += 1
        if label == 2 and fea >= threshold:
            intersection += 1
    print(intersection, union)
    return intersection * 1.0 / union


def calculateThreshold(trainData):
    """
    trainData: type: list, dimension: (N, 2), where the 1st column is scaled 
                     variance (0-256), the 2nd column is the label (0, 1, 2)

    return, type, tuple (float, float), (maxIOU, threshold)
    """
    threshold = 0
    maxIOU = 0
    for th in range(5, 131, 5):
        curIOU = getIOU(trainData, th)
        print('threshold: %d\t IOU: %f' % (th, curIOU))
        if curIOU > maxIOU:
            maxIOU = curIOU
            threshold = th
    return maxIOU, threshold


def getHashCodeSet(hashcodeFilePath):
    """
    hashcodeFilePath: type: str, the path for the hasecode file (train.txt or test.txt)

    return, type, set, a set of the hashcodes
    """
    hashcodeSet = set()
    with open(hashcodeFilePath, 'r') as f:
        for line in f:
            hashcodeSet.add(line.strip())
    return hashcodeSet


def getTrainData(args):
    """
    generate the traing data

    @param: args, parsed arguments
    return: list of list, dimension: (N, 2), where the 1st column is scaled 
                     variance (0-256), the 2nd column is the label (0, 1, 2)
    """
    trainData = []
    hashcodeSet = getHashCodeSet(args.hashcodeFilePath)
    dataDirs = glob(args.dataDirPath + '*')

    cnt = 1
    for direc in dataDirs:
        hashcode = direc.split('/')[-1]
        if hashcode not in hashcodeSet:
            continue
        print(cnt, hashcode)
        cnt += 1

        maskPath = os.path.join(args.maskDirPath, hashcode + '.png')
        curFrame = cv2.imread(os.path.join(direc, 'frame0000.png'), 0)
        row, col = curFrame.shape
        
        hsv = np.zeros((row, col, 3), np.uint8)
        hsv[...,1] = 255
        total = np.zeros((row, col), np.uint8)

        for i in range(1, 100):
            preFrame = curFrame
            curFrame = cv2.imread(os.path.join(direc, 'frame0' + str(i).zfill(3) + '.png'), 0)
            
            flow = cv2.calcOpticalFlowFarneback(preFrame, curFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)        
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

            for i in range(row):
                for j in range(col):
                    if gray[i][j] >= 128:
                        total[i][j] += 2
                    elif gray[i][j] >= 32:
                        total[i][j] += 1
        
        total = total / np.max(total) * 256    
        mask = cv2.imread(maskPath, 0)

        for i in range(row):
            for j in range(col):
                trainData.append((total[i][j], mask[i][j]))

    return trainData


def generateTestResult(threshold, args):
    """
    gerenerate the testing results
    
    @param: threshold: type, float, the threshold of the scaled variance
    @param: args, parsed arguments
    return: void
    """
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    hashcodeSet = getHashCodeSet(args.testHashcodeFilePath)
    dataDirs = glob(args.dataDirPath + '*')

    cnt = 1
    for direc in dataDirs:
        hashcode = direc.split('/')[-1]
        if hashcode not in hashcodeSet:
            continue
        print(cnt, hashcode)
        cnt += 1

        curFrame = cv2.imread(os.path.join(direc, 'frame0000.png'), 0)
        row, col = curFrame.shape
        
        hsv = np.zeros((row, col, 3), np.uint8)
        hsv[...,1] = 255
        total = np.zeros((row, col), np.uint8)

        for i in range(1, 100):
            preFrame = curFrame
            curFrame = cv2.imread(os.path.join(direc, 'frame0' + str(i).zfill(3) + '.png'), 0)
            
            flow = cv2.calcOpticalFlowFarneback(preFrame, curFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)        
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

            for i in range(row):
                for j in range(col):
                    if gray[i][j] >= 128:
                        total[i][j] += 2
                    elif gray[i][j] >= 32:
                        total[i][j] += 1
        
        total = total / np.max(total) * 256    
        
        output = np.zeros_like(total, np.uint8)
        for i in range(row):
            for j in range(col):
                if total[i][j] >= threshold:
                    output[i][j] = 2

        outputImage = Image.fromarray(output)
        outputImage.save(os.path.join(args.output, hashcode + '.png'), 0)


def main(args):
    if args.train:
        trainData = getTrainData(args)
        IOU, threshold = calculateThreshold(trainData)
        print('Threshold is %f' % threshold)
        print(IOU)
    else:
        print('haha')
        threshold = int(args.threshold)

    generateTestResult(threshold, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixel Variance for Cilia Segmentation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-tr', '--train', action='store_true', help='Whether or not to train')
    group.add_argument('-th', '--threshold', default=50, help='Threshold for the pixel variance')
    parser.add_argument('-d', '--dataDirPath', default='/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/data/data/', help='Path for the traing hashcode file, e.g. train.txt')
    parser.add_argument('-m', '--maskDirPath', default='/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/masks', help='Path for the masks directory')
    parser.add_argument('-ha', '--hashcodeFilePath', default='/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/train.txt', help='Path for the traing hashcode file, e.g. train.txt')
    parser.add_argument('-t', '--testHashcodeFilePath', default='/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/test.txt', help='Path for the test hashcode file, e.g. test.txt')
    parser.add_argument('-o', '--output', default="output", help='Path for the output directory')

    args = parser.parse_args()
    main(args)
