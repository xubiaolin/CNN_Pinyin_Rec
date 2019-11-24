import cv2
import numpy as np
from matplotlib import pyplot as plot
import os
import json
import copy
import time


def cut(img):
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
    img = copy.deepcopy(img)
    img = np.asarray(img)
    img[img > 127] = 255
    img[img <= 127] = 1
    img[img == 255] = 0

    cols = np.sum(img, 0)
    rows = np.sum(img, 1)
    # print(cols)

    x1 = 0
    y1 = 0
    y2, x2 = img.shape
    y2 -= 1
    x2 -= 1

    for i in range(len(cols)):
        if cols[i] > 0:
            x1 = i
            break

    for i in range(len(cols)):
        if cols[len(cols) - 1 - i] > 0:
            x2 =len(cols) - 1 - i
            break

    for i in range(len(rows)):
        if rows[i] > 0:
            y1 = i
            break

    for i in range(len(rows)):
        if rows[len(rows) - 1 - i] > 0:
            y2 = len(rows) - 1 - i
            break

    return x1,x2,y1,y2


if __name__ == '__main__':

    path='C:\\Users\\MarkXu\\Desktop\\dst\\shendiao/'
    for i in os.listdir(path):
        char=i
        pre_name = str(int(i)+26)+'_'
        #pre_name =str(ord(char)-ord("a")+1)+"_"
        imgPath = path+char+'/'
        count=0
        savaPath = 'E:\PythonProject\CNN_Pinyin\Data\originData/'+char+'/'
        if not os.path.exists(savaPath):
            os.mkdir(savaPath)
        for i in os.listdir(imgPath):
            img = cv2.imread(imgPath + i, 0)

            x1,x2,y1,y2=cut(img)
            if x1==x2 or y1==y2:
                continue
            print(x1,x2,y1,y2)
            temp=img[y1:y2,x1:x2]
            temp=cv2.resize(temp,(28,28))

            cv2.imwrite(savaPath+pre_name+str(count)+'.jpg',temp)
            count+=1

    # count=1
    # imgPath='C:\\Users\\MarkXu\\Desktop\\test/'
    # for i in os.listdir(imgPath):
    #     img = cv2.imread(imgPath + i, 0)
    #
    #     x1, x2, y1, y2 = cut(img)
    #     if x1 == x2 or y1 == y2:
    #         continue
    #     print(x1, x2, y1, y2)
    #     temp = img[y1:y2, x1:x2]
    #     temp = cv2.resize(temp, (28, 28))
    #
    #     cv2.imwrite(imgPath + i , temp)
    #     count += 1