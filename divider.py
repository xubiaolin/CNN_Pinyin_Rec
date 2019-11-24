import cv2
import numpy as np
import copy
import operator
import os
import cv2
import train_and_val
from datetime import datetime

counterlist={}
index=[]
yzuobiao=[]
xzuobiao=[]
dic={}
finalimglist=[]
finalimglist1=[]
xindex=[]
dic1={}
def shuipingtouying(img):

    img = np.asarray(img)
    height, width = img.shape[:2]
    imgback = copy.deepcopy(img)
    imgback[imgback <= 127] = 1
    imgback[imgback > 127] = 0
    z = np.sum(imgback, 1)


    num_yesorno=False

    listrs=[]
    listre=[]
    for i in range(0,height):
        if num_yesorno==False and z[i]!=0:
            num_yesorno=True
            start_index=i
            listrs.append(start_index)
        else:
            if num_yesorno==True and z[i]==0:
                num_yesorno = False
                end_index=i
                listre.append(end_index)

    return listrs,listre





def chuizhitouying(img):

    img=np.asarray(img)
    height, width = img.shape[:2]
    imgback = copy.deepcopy(img)

    imgback[imgback <= 127] = 1
    imgback[imgback > 127] = 0
    v = np.sum(imgback, 0)

    num_yesorno=False

    listcs=[]
    listce=[]
    for i in range(0,width):
        if num_yesorno==False and v[i]!=0:
            num_yesorno=True
            start_index=i
            listcs.append(start_index)
        else:
            if num_yesorno==True and v[i]==0:
                num_yesorno = False
                end_index=i;
                listce.append(end_index)

    return listcs,listce




def pingyinfengge(img):
    i=0
    yindex=[]

    imglist=[]

    listcs=[]
    # imgffbb=img.copy()
    # img1 = cv2.cvtColor(imgffbb, cv2.COLOR_BGR2GRAY)


    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img1 = cv2.erode(img1, kernel)

    height, width = img1.shape[:2]
    listcs=chuizhitouying(img1)[0]
    listce = chuizhitouying(img1)[1]
    zuobianju=listcs[0]
    if len(listcs)>len(listce):
        listce.append(width)

    for m in range(0,len(listcs)):
        height, width = img.shape[:2]
        img2 = img[0:height, listcs[m]:listce[m]]#3通道


        img21 = img1[0:height, listcs[m]:listce[m]]#1通道


        img2fuben=img2.copy()#3通道
        img2fuben1 = img2.copy()  # 3通道
        heightimg2, widthimg2 = img2.shape[:2]
        ret, binary = cv2.threshold(img21, 127, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)







        if len(contours) > 1:
            xmaxlist = []
            xminlist = []
            ymaxlist = []
            yminlist = []
            for j in range(0, len(contours)):
                xmax = contours[j][0][0][0]
                ymax = contours[j][0][0][1]
                xmin = contours[j][0][0][0]
                ymin = contours[j][0][0][1]

                for k in range(0, len(contours[j])):
                    if xmax < contours[j][k][0][0] :
                        xmax = contours[j][k][0][0]
                    if ymax < contours[j][k][0][1]:
                        ymax = contours[j][k][0][1]
                    if xmin > contours[j][k][0][0]:
                        xmin = contours[j][k][0][0]
                    if ymin > contours[j][k][0][1]:
                        ymin = contours[j][k][0][1]
                xmaxlist.append(xmax)
                ymaxlist.append(ymax)
                xminlist.append(xmin)
                yminlist.append(ymin)

            flag=0
            for j in range(0, len(xmaxlist)):
                for k in range(j+1 , len(xmaxlist)):
                    if xmaxlist[j] <= xmaxlist[k] and ymaxlist[j] <= ymaxlist[k] and xminlist[j] <= xminlist[k] and yminlist[j] <= yminlist[k]:
                        id = j
                    elif xmaxlist[k] <= xmaxlist[j] and ymaxlist[k] <= ymaxlist[j] and xminlist[k] >= xminlist[j] and yminlist[k] >= yminlist[j]:
                        id = k
                    else:
                        id=None
                    if id is not None:
                        del contours[id-flag]
                        flag=flag+1









        if widthimg2 > 1/2* (width-2*zuobianju):
            index=[]
            xx=9999
            for j in range(0,len(contours)):
                c = contours[j]
                for j in range(0,len(c)):
                    if xx>c[j][0][0]:
                        xx=c[j][0][0]

                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))

                for j in range(0,len(box)):
                    if box[j][0]<0:
                        box[j][0]=0
                    if box[j][1]<0:
                        box[j][1] = 0

                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs)
                x2 = max(Xs)
                y1 = min(Ys)
                y2 = max(Ys)
                hight = y2 - y1
                width = x2 - x1
                crop_img = img21[y1:y1 + hight, x1:x1 + width]


                crop_imgfuben=img2fuben[y1:y1 + hight, x1:x1 + width]


                ret, binary = cv2.threshold(crop_img, 127, 255, cv2.THRESH_BINARY)
                contours1, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




                for j in range(0, len(contours1)):
                    c = contours1[j]
                    counterlist[j]=len(c)
                res = sorted(counterlist.items(), key=lambda asd: asd[1],reverse=True)



                for p in range(0, len(contours1[res[0][0]])):
                        yindex.append(contours1[res[0][0]][p][0][1])


                yindex.sort()



                cankaodian = (yindex[len(yindex) - 1] + yindex[0]) / 2
                cankaodian1 = xx
                yindex.clear()






                for p in range(1,len(res)):
                    index.append(res[p][0])

                if len(contours1)==1:




                    imglist.append(crop_imgfuben)
                    # cv2.imshow("fill_color", crop_imgfuben)
                    # cv2.waitKey(0)

                    yzuobiao.append(cankaodian)


                    xzuobiao.append(cankaodian1)

                else:
                    for k in index:
                        rect = cv2.minAreaRect(contours1[k])
                        box = np.int0(cv2.boxPoints(rect))
                        for j in range(0, len(box)):
                            if box[j][0] < 0:
                                box[j][0] = 0
                            if box[j][1] < 0:
                                box[j][1] = 0
                        Xs = [i[0] for i in box]
                        Ys = [i[1] for i in box]
                        x1 = min(Xs)
                        x2 = max(Xs)
                        y1 = min(Ys)
                        y2 = max(Ys)
                        #cankaodian=(y1+y2)/2
                        img2fuben=img2fuben1.copy()
                        h, w = crop_imgfuben.shape[:2]
                        mask = np.zeros([h + 2, w + 2], np.uint8)

                        cv2.floodFill(crop_imgfuben, mask, tuple(contours1[k][0][0]), (255, 255, 255), (100, 100, 100), (50, 50, 50),cv2.FLOODFILL_FIXED_RANGE)
                        counterlist.clear()
                    # cv2.imshow("fill_color", crop_imgfuben)
                    # cv2.waitKey(0)
                    # cv2.imwrite("D:\\test\\fg6\\"+datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg",crop_imgfuben)
                    imglist.append(crop_imgfuben)
                    yzuobiao.append(cankaodian)
                    print(yzuobiao)
                    xzuobiao.append(cankaodian1)


                counterlist.clear()
                index.clear()


        else:


                height, width = img2fuben.shape[:2]



                img2fubenfuben=img2fuben.copy()
                img2fubenfuben = cv2.cvtColor(img2fubenfuben, cv2.COLOR_BGR2GRAY)

                ret, binary = cv2.threshold(img2fubenfuben, 127, 255, cv2.THRESH_BINARY)

                contours2, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)





                listrs = shuipingtouying(img2fubenfuben)[0]
                listre = shuipingtouying(img2fubenfuben)[1]
                if len(listrs) > len(listre):
                    listre.append(height)


                if len(listrs)==3:
                        crop_img = img2fubenfuben[listrs[0]: listre[0], 0: width]
                        crop_img1=  img2fubenfuben[listrs[1]: listre[2], 0: width]
                        imglist.append(crop_img)
                        yzuobiao.append(0)
                        xzuobiao.append(listcs[m])
                        imglist.append(crop_img1)
                        yzuobiao.append(99999)
                        xzuobiao.append(listcs[m])





                else:
                    if height / width > 3:

                        crop_img = img2fubenfuben[0: height, 0: width]


                        # cv2.imwrite("D:\\test\\fg6\\" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", crop_img)
                        imglist.append(crop_img)
                        yzuobiao.append(99999+i)
                        xzuobiao.append(listcs[m])
                        i=i+1
                    else:
                        if len(contours2)>1:
                            xmaxlist = []
                            xminlist = []
                            ymaxlist = []
                            yminlist = []
                            for j in range(0,len(contours2)):
                                xmax=contours2[j][0][0][0]
                                ymax=contours2[j][0][0][1]
                                xmin = contours2[j][0][0][0]
                                ymin = contours2[j][0][0][1]

                                for k in range(0,len(contours2[j])):
                                    if xmax<contours2[j][k][0][0]:
                                        xmax = contours2[j][k][0][0]
                                    if ymax<contours2[j][k][0][1]:
                                        ymax = contours2[j][k][0][1]
                                    if xmin>contours2[j][k][0][0]:
                                        xmin = contours2[j][k][0][0]
                                    if ymin>contours2[j][k][0][1]:
                                        ymin = contours2[j][k][0][1]
                                xmaxlist.append(xmax)
                                ymaxlist.append(ymax)
                                xminlist.append(xmin)
                                yminlist.append(ymin)
                            flag = 0
                            for j in range(0,len(xmaxlist)):
                                for k in range(j+1,len(xmaxlist)):
                                    if xmaxlist[j]<=xmaxlist[k] and ymaxlist[j]<=ymaxlist[k] and xminlist[j]>=xminlist[k] and yminlist[j]>=yminlist[k]:
                                        index=j
                                    elif xmaxlist[k] <= xmaxlist[j] and ymaxlist[k] <= ymaxlist[j] and xminlist[k] >= xminlist[j] and yminlist[k] >= yminlist[j]:
                                        index = k
                                    else:
                                        index=None

                                    if index is not None:
                                        del contours2[index-flag]
                                        flag=flag+1
                        # print("***********")
                        # print(len(contours))
                        # print("***********")


                        for k in range(0,len(contours2)):
                            for p in range(0,len(contours2[k])):
                                yindex.append(contours2[k][p][0][1])
                                xindex.append(contours2[k][p][0][0])
                            yindex.sort()
                            xindex.sort()
                            cankaodian=(yindex[len(yindex)-1]+yindex[0])/2
                            zengliang=xindex[0]
                            yindex.clear()
                            xindex.clear()




                            rect = cv2.minAreaRect(contours2[k])
                            box = np.int0(cv2.boxPoints(rect))
                            for j in range(0, len(box)):
                                if box[j][0] < 0:
                                    box[j][0] = 0
                                if box[j][1] < 0:
                                    box[j][1] = 0
                            Xs = [i[0] for i in box]
                            Ys = [i[1] for i in box]
                            x1 = min(Xs)
                            x2 = max(Xs)
                            y1 = min(Ys)
                            y2 = max(Ys)
                            hight = y2 - y1
                            width = x2 - x1
                            crop_img = img2fubenfuben[y1:y1 + hight, x1:x1 + width]



                            # cv2.imwrite("D:\\test\\fg6\\" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", crop_img)

                            imglist.append(crop_img)
                            yzuobiao.append(cankaodian)
                            xzuobiao.append(listcs[m]+zengliang)


    for i in range(0,len(yzuobiao)):
        dic[yzuobiao[i]]=[imglist[i],xzuobiao[i]]



    for k in sorted(dic,reverse=True):

        finalimglist.append(dic[k])



    shengdiaoimg=finalimglist[len(finalimglist)-1][0]
    del finalimglist[len(finalimglist)-1]





    for j in range(0,len(finalimglist)):
        if len(finalimglist[j][0])<15:
                continue
        else:
            dic1[finalimglist[j][1]] = finalimglist[j][0]






    for k in sorted(dic1.keys()):

        finalimglist1.append(dic1[k])
    finalimglist1.append(shengdiaoimg)



    return finalimglist1


def getdivide(imgpath):
    img=cv2.imread(imgpath)
    imglist=pingyinfengge(img)
    for i in range(len(imglist)):
        temp=imglist[i]
        temp=cv2.resize(temp,(25,25))
        img1 = cv2.copyMakeBorder(temp, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        imglist[i]=img1
    return imglist
if __name__ == '__main__':
    for i in os.listdir('C:\\Users\\MarkXu\\Desktop\\dst\\chars/'):

        dict1='C:\\Users\\MarkXu\\Desktop\\dst\\chars/'+i
        for j in os.listdir(dict1):
            img = cv2.imread(dict1+'/'+j)
            cv2.imshow("img",img)
            cv2.waitKey(100)
            imglist=pingyinfengge(img)#参数为你想裁剪的图片
            count=0
            for j in imglist:#显示试试
                cv2.imshow("temp",j)
                cv2.waitKey(0)

                count+=1
