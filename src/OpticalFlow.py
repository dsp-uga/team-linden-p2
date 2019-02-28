import cv2
import os
import numpy as np
from glob import glob
from PIL import Image


ISTRAIN = True
THRESHOLD = 50

DATADIRSPATH = '/Users/user/Documents/edu/courses/csci8360/project2/dat/data/*' #'/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/data/data/*'
MASKDIRPATH = '/Users/user/Documents/edu/courses/csci8360/project2/dat/masks' #'/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/masks'
HASHCODEFILEPATH = '/Users/user/Documents/edu/courses/csci8360/project2/dat/train.txt' #'/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/train.txt'
TESTHASHCODEFILEPATH = '/Users/user/Documents/edu/courses/csci8360/project2/dat/test.txt' #'/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p2/project2/test.txt'
OUTPUTDIR = '/Users/user/Documents/edu/courses/csci8360/project2/dat/output'



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


def getTrainData():
    """
    generate the traing data

    return: list of list, dimension: (N, 2), where the 1st column is scaled 
                     variance (0-256), the 2nd column is the label (0, 1, 2)
    """
    trainData = []
    hashcodeSet = getHashCodeSet(HASHCODEFILEPATH)
    dataDirs = glob(DATADIRSPATH)

    cnt = 1
    for direc in dataDirs:
        hashcode = direc.split('/')[-1]
        if hashcode not in hashcodeSet:
            continue
        print(cnt, hashcode)
        cnt += 1

        maskPath = os.path.join(MASKDIRPATH, hashcode + '.png')
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


def generateTestResult(threshold):
    """
    gerenerate the testing results
    
    threshold: type, float, the threshold of the scaled variance
    return: void
    """
    if not os.path.exists(OUTPUTDIR):
        os.makedirs(OUTPUTDIR)

    hashcodeSet = getHashCodeSet(TESTHASHCODEFILEPATH)
    dataDirs = glob(DATADIRSPATH)

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
        outputImage.save(os.path.join(OUTPUTDIR, hashcode + '.png'), 0)


def main():
    if ISTRAIN:
        trainData = getTrainData()
        IOU, threshold = calculateThreshold(trainData)
        print('Threshold is %f' % threshold)
        print(IOU)
    else:
        threshold = THRESHOLD

    generateTestResult(threshold)


if __name__ == '__main__':
    main()
