import cv2
import numpy as np
from matplotlib import pyplot as plot
import os

def cut(img):

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

    img = np.asarray(img)
    img[img > 127] = 255
    img[img <= 127] = 1
    img[img == 255] = 0

    cols = np.sum(img, 0)
    rows = np.sum(img, 1)

    r, c = img.shape
    cpoint = []
    for i in range(len(cols)):
        if cols[i] > r - 400:
            cpoint.append(i)

    x1 = []
    x2 = []
    for i in range(len(cpoint) - 1):
        if cpoint[i + 1] - cpoint[i] > 10:
            x1.append(cpoint[i])
            x2.append(cpoint[i + 1])

    rpoint = []
    for i in range(len(rows)):
        if rows[i] > c - 400:
            rpoint.append(i)

    y1 = []
    y2 = []
    for i in range(len(rpoint) - 1):
        if rpoint[i + 1] - rpoint[i] > 10:
            y1.append(rpoint[i])
            y2.append(rpoint[i + 1])

    return x1, x2, y1, y2


if __name__ == '__main__':
    preName = '26_'
    preFile='z'
    imgPath = 'C:/Users/MarkXu/Desktop/dataset/'+preFile+'.jpg'
    savaPath='Data/originData/'+preFile+'/'

    if not os.path.exists(savaPath):
        os.mkdir(savaPath)
    img = cv2.imread(imgPath, 0)

    x1,x2,y1,y2=cut(img)
    img=np.asarray(img)
    img[img>127]=255
    img[img<=127]=0

    count=len(os.listdir(savaPath))
    if len(os.listdir(savaPath)) ==0:
        pass
    else:
        preName=os.listdir(savaPath)[0].split('_')[0]+'_'
    for i,j in zip(x1,x2):
        for m,n in zip(y1,y2):
            # cv2.imshow("img",img[m:n,i:j])
            # cv2.waitKey(500)
            print(savaPath+preName+str(count)+".jpg")
            temp=img[m:n,i:j]
            temp=cv2.resize(temp,(28,28))
            cv2.imwrite(savaPath+preName+str(count)+".jpg",temp)
            count+=1