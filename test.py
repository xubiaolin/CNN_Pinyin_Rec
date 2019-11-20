import numpy as np
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

CHAR_SET_LEN=26

# 文本转向量
def text2vec(text):
    text_len = len(text)
    vector = np.zeros(1 * CHAR_SET_LEN)
    vector[int(text)]=1
    return vector



# 读MNIST数据集的图片数据
def mnist_load_img(img_path):
    with open(img_path, "rb") as fp:
        # >是以大端模式读取，i是整型模式，读取前四位的标志位，
        # unpack()函数：是将4个字节联合后再解析成一个数，(读取后指针自动后移)
        msb = struct.unpack('>i', fp.read(4))[0]
        # 标志位为2051，后存图像数据；标志位为2049，后存图像标签
        if msb == 2051:
            # 读取样本个数60000，存入cnt
            cnt = struct.unpack('>i', fp.read(4))[0]
            # rows：行数28；cols：列数28
            rows = struct.unpack('>i', fp.read(4))[0]
            cols = struct.unpack('>i', fp.read(4))[0]
            imgs = np.empty((cnt, rows, cols), dtype="int")
            for i in range(0, cnt):
                for j in range(0, rows):
                    for k in range(0, cols):
                        # 16进制转10进制
                        pxl = int(hex(fp.read(1)[0]), 16)
                        imgs[i][j][k] = pxl
            return imgs
        else:
            return np.empty(1)

def mnist_plot_img(img):
    (rows, cols) = img.shape;
    plt.figure();
    plt.gray();
    plt.imshow(img);
    plt.show();


def pengzhangtupian(img):
    img=copy.deepcopy(img)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel)
    return erosion

if __name__ == '__main__':
   path='Data/originData/y/'
   count=len(os.listdir(path))
   pre=os.listdir(path)[0].split('_')[0]+'_'

   for i in os.listdir(path):
       img=cv2.imread(path+i,0)
       img=pengzhangtupian(img)
       dstName=path+pre+str(count)+'.jpg'
       print(dstName)
       cv2.imwrite(dstName,img)
       count+=1

