import cv2
import os
import numpy as np
from glob import glob
from PIL import Image

RAWTESTPATH = '/Users/jerryhui/Downloads/p2/*'
OUTPUTDIR = '/Users/jerryhui/Downloads/p2BoostTiramisu/'

def main():
    fileList = glob(RAWTESTPATH)
    for file in fileList:
        image = cv2.imread(file, 0)
        row, col = image.shape
        output = np.zeros_like(image, np.uint8)

        for i in range(row):
            for j in range(col):
                isCilia = False
                for x in range(i - 2, i + 3):
                    for y in range(j - 2, j + 3):
                        if 0 <= x < row and 0 <= y < col and image[x][y] == 2:
                            output[i][j] = 2
                            isCilia = True
                            break
                    if isCilia:
                        break
        # print(np.sum(image))
        # print(np.sum(output))
        # break
        hashcode = file.split('/')[-1]
        outputImage = Image.fromarray(output)
        outputImage.save(os.path.join(OUTPUTDIR, hashcode), 0)


if __name__ == '__main__':
    main()


