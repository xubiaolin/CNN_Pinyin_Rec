import cv2
import numpy as np
from matplotlib import pyplot as plot
import os
import json
import copy
import time
import shutil
from dataUtils import shutildata


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
            x2 = len(cols) - 1 - i
            break

    for i in range(len(rows)):
        if rows[i] > 0:
            y1 = i
            break

    for i in range(len(rows)):
        if rows[len(rows) - 1 - i] > 0:
            y2 = len(rows) - 1 - i
            break

    return x1, x2, y1, y2


def resize(img):
    width, height = img.shape
    if width > height:
        wider = (width - height) // 2
        img=cv2.copyMakeBorder(img,0,0,wider,wider,cv2.BORDER_CONSTANT, value=[255, 255, 255])

    elif height>width:
        wider=(height - width)//2
        img=cv2.copyMakeBorder(img, wider, wider,0,0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img

# a-z
def makeOringinData(path):
    datapath="E:\\PythonProject\\CNN_Pinyin/Data/originData/"
    if os.path.exists(datapath):
        shutil.rmtree(datapath)

    os.mkdir(datapath)

    with open('num_char.json', 'r') as f:
        dict = json.loads(f.read())

    for i in os.listdir(path):
        char = i
        pre_name=dict[str(char)]+'_'
        print(pre_name)
        imgPath = path + char + '/'
        count = 0
        savaPath = datapath+ char + '/'
        print(savaPath)

        dirlist=os.listdir(imgPath)
        if len(dirlist) == 0:
            continue

        if os.path.exists(savaPath):
            shutil.rmtree(savaPath)

        if not os.path.exists(savaPath):
            os.mkdir(savaPath)

        for i in dirlist:
            img = cv2.imread(imgPath + i, 0)
            temp=img
            print(i)
            try:
                width,height=img.shape
            except:
                continue
            if not width==height:
                x1, x2, y1, y2 = cut(img)
                if x1 == x2 or y1 == y2:
                    continue
                print(x1, x2, y1, y2)
                temp = img[y1:y2, x1:x2]
                temp=resize(temp)
            temp = cv2.resize(temp, (28, 28))

            cv2.imwrite(savaPath + pre_name + str(count) + '.jpg', temp)
            count += 1


if __name__ == '__main__':

    path = 'C:\\Users\\MarkXu\\Desktop\\dst\\chars/'
    makeOringinData(path)
    shutildata('Data/originData/', 'Data/train/', 'Data/test/')



