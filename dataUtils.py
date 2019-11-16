# 随机分拣出测试集，其他文件为训练集
import random
import os
import shutil
import cv2

def shutildata(originData, trainpath, testpath):
    if not os.path.exists(trainpath):
        os.mkdir(trainpath)
    if not os.path.exists(testpath):
        os.mkdir(testpath)

    parent = os.listdir(originData)
    for i in parent:
        child = os.listdir(originData + i)
        index = [random.randint(0, len(child)-1) for i in range(10)]
        arr = [child[i].split('.')[0].split("_")[1] for i in index]
        print(arr)

        for j in child:
            if j.split('.')[0].split('_')[1] in arr:
                shutil.copy(originData + i + "/" + j, testpath)

            else:
                shutil.copy(originData + i + '/' + j, trainpath)


def rename():
    path = 'C:\\Users\\MarkXu\\Desktop\\xiaoxie\\'
    for i in os.listdir(path):


        count = len(os.listdir('Data/originData/'+chr(int(i)+ord('a')-1)+'/'))
        for j in os.listdir(path+i):
            print(path+i+'/'+j)
            img=cv2.imread(path+i+'/'+j,0)
            img=cv2.resize(img,(28,28))
            name=int(i)
            os.remove(path+i+'/'+j)
            cv2.imwrite(path+i+'/'+str(name)+'_'+str(count)+'.jpg',img)
            count+=1



if __name__ == '__main__':
    shutildata('Data/originData/', 'Data/train/', 'Data/test/')
    #rename()