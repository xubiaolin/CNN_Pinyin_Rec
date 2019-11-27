import cv2
import numpy as np
import copy
import operator
import os
import cv2
import numpy as np
import copy
import operator

import cv2
from datetime import datetime


def shuipingtouying(img):
    img = np.asarray(img)
    height, width = img.shape[:2]
    imgback = copy.deepcopy(img)
    imgback[imgback <= 127] = 1
    imgback[imgback > 127] = 0
    z = np.sum(imgback, 1)

    num_yesorno = False

    listrs = []
    listre = []
    for i in range(0, height):
        if num_yesorno == False and z[i] != 0:
            num_yesorno = True
            start_index = i
            listrs.append(start_index)
        else:
            if num_yesorno == True and z[i] == 0:
                num_yesorno = False
                end_index = i
                listre.append(end_index)

    return listrs, listre


def chuizhitouying(img):
    img = np.asarray(img)
    height, width = img.shape[:2]
    imgback = copy.deepcopy(img)

    imgback[imgback <= 127] = 1
    imgback[imgback > 127] = 0
    v = np.sum(imgback, 0)

    num_yesorno = False

    listcs = []
    listce = []
    for i in range(0, width):
        if num_yesorno == False and v[i] != 0:
            num_yesorno = True
            start_index = i
            listcs.append(start_index)
        else:
            if num_yesorno == True and v[i] == 0:
                num_yesorno = False
                end_index = i;
                listce.append(end_index)

    return listcs, listce

def pinyinfenge(img):
        imglist=[]
        zuobiao={}
        zuobiao1 = {}
        dic={}
        yindex=[]
        idic={}
        iyindex=[]
        index=[]
        kxindex=[]
        paixu={}
        kxindex1 = []
        paixu1 = {}
        #img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1=img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img1 = cv2.erode(img1, kernel)

        height, width = img1.shape[:2]
        listcs=chuizhitouying(img1)[0]
        listce = chuizhitouying(img1)[1]

        if len(listcs) > len(listce):
            listce.append(width)


        ret, binary = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)

        icontours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for k in range(0, len(icontours)):
            for p in range(0, len(icontours[k])):
                iyindex.append(icontours[k][p][0][1])
            iyindex.sort()
            icankaodian = (iyindex[len(iyindex) - 1] + iyindex[0]) / 2
            idic[k] = icankaodian
            iyindex.clear()
        ires = sorted(idic.items(), key=lambda asd: asd[1], reverse=True)
        yindiao=ires[len(ires)-1][1]






        kuan=width-listcs[0]-(width-listce[len(listce)-1])
        for m in range(0,len(listcs)):
            height, width = img.shape[:2]

            img2 = img[0:height, listcs[m]:listce[m]]#3通道

            img21 = img1[0:height, listcs[m]:listce[m]]#1通道


            listrs = shuipingtouying(img21)[0]
            listre = shuipingtouying(img21)[1]
            zuobianju = listcs[0]
            if len(listrs) > len(listre):
                listre.append(height)




            img2fuben=img2.copy()#3通道
            img2fuben1 = img21.copy()  # 3通道
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



            if widthimg2 > 1/2* kuan-20 and len(contours)>1:
                print("特殊情况")

                for k in range(0, len(contours)):
                    for p in range(0, len(contours[k])):
                        yindex.append(contours[k][p][0][1])

                    yindex.sort()
                    cankaodian = (yindex[len(yindex) - 1] + yindex[0]) / 2
                    dic[k]=cankaodian
                    yindex.clear()

                res = sorted(dic.items(), key=lambda asd: asd[1], reverse=True)


                rect = cv2.minAreaRect(contours[res[len(res)-1][0]])
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
                img2fuben=img21.copy()
                img2fuben1=img21.copy()


                h, w = img2fuben.shape[:2]

                mask = np.zeros([h + 2, w + 2], np.uint8)


                cv2.floodFill(img2fuben, mask, tuple(contours[res[len(res)-1][0]][1][0]), (255, 255, 255), (100, 100, 100),
                              (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)

                img2fuben = cv2.medianBlur(img2fuben, 3)


                ilistcs = chuizhitouying(img2fuben)[0]
                ilistce = chuizhitouying(img2fuben)[1]

                if len(ilistcs) > len(ilistce):
                    ilistce.append(widthimg2)







                for j in range(0,len(ilistcs)):
                    if len(ilistcs)==1:


                        for o in range(0,len(res)-1):
                            index.append(res[o][0])


                        if res[len(res)-1][1]==ires[len(ires)-1][1] and ires[len(ires)-1][1]<h/3:




                                img2fuben10 = img2fuben1.copy()



                                h, w = img2fuben1.shape[:2]
                                mask = np.zeros([h + 2, w + 2], np.uint8)

                                cv2.floodFill(img2fuben10, mask, tuple(contours[res[len(res)-1][0]][0][0]), (255, 255, 255),
                                              (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)





                                for k in index:
                                    for p in range(0, len(contours[k])):
                                        kxindex.append(contours[k][p][0][0])
                                    kxindex.sort()
                                    # paixu[k]=(kxindex[0],kxindex[len(kxindex)-1])
                                    paixu[k] = kxindex[0]
                                    zuobiao[kxindex[0]]=kxindex[len(kxindex)-1]
                                    kxindex.clear()

                                paixures = sorted(paixu.items(), key=lambda asd: asd[0], reverse=True)




                                for j in range(0,len(paixures)):
                                    if j==0:
                                        fgimg=img2fuben10[0:h,paixures[j][1]:zuobiao[paixures[j][1]]]

                                        ret, binary = cv2.threshold(fgimg, 127, 255, cv2.THRESH_BINARY)
                                        contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE,
                                                                               cv2.CHAIN_APPROX_SIMPLE)


                                        oxmaxlist = []
                                        oxminlist = []
                                        oymaxlist = []
                                        oyminlist = []
                                        for j in range(0, len(contours)):
                                            xmax = contours[j][0][0][0]
                                            ymax = contours[j][0][0][1]
                                            xmin = contours[j][0][0][0]
                                            ymin = contours[j][0][0][1]

                                            for k in range(0, len(contours[j])):
                                                if xmax < contours[j][k][0][0]:
                                                    xmax = contours[j][k][0][0]
                                                if ymax < contours[j][k][0][1]:
                                                    ymax = contours[j][k][0][1]
                                                if xmin > contours[j][k][0][0]:
                                                    xmin = contours[j][k][0][0]
                                                if ymin > contours[j][k][0][1]:
                                                    ymin = contours[j][k][0][1]
                                            oxmaxlist.append(xmax)
                                            oymaxlist.append(ymax)
                                            oxminlist.append(xmin)
                                            oyminlist.append(ymin)

                                        flag = 0
                                        for j in range(0, len(oxmaxlist)):
                                            for k in range(j + 1, len(oxmaxlist)):
                                                if oxmaxlist[j] <= oxmaxlist[k] and oymaxlist[j] <= oymaxlist[k] and \
                                                        oxminlist[j] <= oxminlist[k] and oyminlist[j] <= oyminlist[k]:
                                                    id = j
                                                elif oxmaxlist[k] <= oxmaxlist[j] and oymaxlist[k] <= oymaxlist[j] and \
                                                        oxminlist[k] >= oxminlist[j] and oyminlist[k] >= oyminlist[j]:
                                                    id = k
                                                else:
                                                    id = None
                                                if id is not None:
                                                    del contours[id - flag]
                                                    flag = flag + 1
                                        oxmaxlist.clear()
                                        oxminlist.clear()
                                        oymaxlist.clear()
                                        oyminlist.clear()


                                        minllencontours=1000000000
                                        for k in range(0,len(contours)):
                                            if minllencontours>len(contours[k]):
                                                op=k
                                                minllencontours=len(contours[k])
                                        print(op)
                                        rect = cv2.minAreaRect(contours[op])
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


                                        h, w = fgimg.shape[:2]
                                        mask = np.zeros([h + 2, w + 2], np.uint8)

                                        cv2.floodFill(fgimg, mask, tuple(contours[op][1][0]), (255, 255, 255),
                                                      (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                                        fgimg = cv2.medianBlur(fgimg, 3)
                                        imglist.append(fgimg)

                                        # cv2.imshow("fill_color", fgimg)
                                        # cv2.waitKey(0)
                                    else:

                                        fgimg = img2fuben1[0:h, paixures[j][1]:zuobiao[paixures[j][1]]]

                                        ret, binary = cv2.threshold(fgimg, 127, 255, cv2.THRESH_BINARY)
                                        fcontours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE,
                                                                               cv2.CHAIN_APPROX_SIMPLE)

                                        oxmaxlist = []
                                        oxminlist = []
                                        oymaxlist = []
                                        oyminlist = []
                                        for j in range(0, len(fcontours)):
                                            xmax = fcontours[j][0][0][0]
                                            ymax = fcontours[j][0][0][1]
                                            xmin = fcontours[j][0][0][0]
                                            ymin = fcontours[j][0][0][1]

                                            for k in range(0, len(fcontours[j])):
                                                if xmax < fcontours[j][k][0][0]:
                                                    xmax = fcontours[j][k][0][0]
                                                if ymax < fcontours[j][k][0][1]:
                                                    ymax = fcontours[j][k][0][1]
                                                if xmin > fcontours[j][k][0][0]:
                                                    xmin = fcontours[j][k][0][0]
                                                if ymin > fcontours[j][k][0][1]:
                                                    ymin = fcontours[j][k][0][1]
                                            oxmaxlist.append(xmax)
                                            oymaxlist.append(ymax)
                                            oxminlist.append(xmin)
                                            oyminlist.append(ymin)

                                        flag = 0
                                        for j in range(0, len(oxmaxlist)):
                                            for k in range(j + 1, len(oxmaxlist)):
                                                if oxmaxlist[j] <= oxmaxlist[k] and oymaxlist[j] <= oymaxlist[k] and \
                                                        oxminlist[j] <= oxminlist[k] and oyminlist[j] <= oyminlist[k]:
                                                    id = j
                                                elif oxmaxlist[k] <= oxmaxlist[j] and oymaxlist[k] <= oymaxlist[j] and \
                                                        oxminlist[k] >= oxminlist[j] and oyminlist[k] >= oyminlist[j]:
                                                    id = k
                                                else:
                                                    id = None
                                                if id is not None:
                                                    del fcontours[id - flag]
                                                    flag = flag + 1
                                        oxmaxlist.clear()
                                        oxminlist.clear()
                                        oymaxlist.clear()
                                        oyminlist.clear()







                                        minllencontours = 1000000000

                                        for k in range(0, len(fcontours)):

                                            if minllencontours > len(fcontours[k]):
                                                op = k
                                                minllencontours = len(fcontours[k])



                                        h, w = fgimg.shape[:2]
                                        mask = np.zeros([h + 2, w + 2], np.uint8)

                                        cv2.floodFill(fgimg, mask, tuple(fcontours[op][0][0]), (255, 255, 255),
                                                      (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
                                        fgimg = cv2.medianBlur(fgimg, 3)
                                        imglist.append(fgimg)

                                        # cv2.imshow("fill_color", fgimg)
                                        # cv2.waitKey(0)




                        else:#直接选
                            print("hghgggygyg")
                            ret, binary = cv2.threshold(img2fuben1, 127, 255, cv2.THRESH_BINARY)

                            contours, hierarchy = cv2.findContours(~binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            oxmaxlist = []
                            oxminlist = []
                            oymaxlist = []
                            oyminlist = []
                            for j in range(0, len(contours)):
                                xmax = contours[j][0][0][0]
                                ymax = contours[j][0][0][1]
                                xmin = contours[j][0][0][0]
                                ymin = contours[j][0][0][1]

                                for k in range(0, len(contours[j])):
                                    if xmax < contours[j][k][0][0]:
                                        xmax = contours[j][k][0][0]
                                    if ymax < contours[j][k][0][1]:
                                        ymax = contours[j][k][0][1]
                                    if xmin > contours[j][k][0][0]:
                                        xmin = contours[j][k][0][0]
                                    if ymin > contours[j][k][0][1]:
                                        ymin = contours[j][k][0][1]
                                oxmaxlist.append(xmax)
                                oymaxlist.append(ymax)
                                oxminlist.append(xmin)
                                oyminlist.append(ymin)

                            flag = 0
                            for j in range(0, len(oxmaxlist)):
                                for k in range(j + 1, len(oxmaxlist)):
                                    if oxmaxlist[j] <= oxmaxlist[k] and oymaxlist[j] <= oymaxlist[k] and \
                                            oxminlist[j] <= oxminlist[k] and oyminlist[j] <= oyminlist[k]:
                                        id = j
                                    elif oxmaxlist[k] <= oxmaxlist[j] and oymaxlist[k] <= oymaxlist[j] and \
                                            oxminlist[k] >= oxminlist[j] and oyminlist[k] >= oyminlist[j]:
                                        id = k
                                    else:
                                        id = None
                                    if id is not None:
                                        del contours[id - flag]
                                        flag = flag + 1
                            oxmaxlist.clear()
                            oxminlist.clear()
                            oymaxlist.clear()
                            oyminlist.clear()
                            print(len(contours))

                            for k in range(0,len(contours)):
                                for p in range(0, len(contours[k])):
                                    kxindex1.append(contours[k][p][0][0])
                                kxindex1.sort()
                                # paixu[k]=(kxindex[0],kxindex[len(kxindex)-1])
                                paixu1[k] = kxindex1[0]
                                zuobiao1[kxindex1[0]] = kxindex1[len(kxindex1) - 1]
                                kxindex1.clear()

                            paixures1 = sorted(paixu1.items(), key=lambda asd: asd[0], reverse=True)
                            for j in range(0,len(paixures1)-1):
                                if abs(paixures1[j][1]-paixures1[j+1][1])<12:
                                    if paixures1[j+1][1]-paixures1[j][1]>0:
                                        del paixures1[j]
                                    else:
                                        del paixures1[j+1]
                            h, w = img2fuben1.shape[:2]

                            for k in range(0,len(paixures1)):
                                fgimg = img2fuben1[0:h, paixures1[k][1]:zuobiao1[paixures1[k][1]]]
                                fgimg = cv2.medianBlur(fgimg, 3)
                                imglist.append(imgfg)
                                # cv2.imshow("sga",fgimg)
                                # cv2.waitKey(0)











                    elif j==0:
                        imgfg = img2fuben1[0:heightimg2, ilistcs[j]:ilistce[j]]  # 3通道
                        imglist.append(imgfg)
                        # cv2.imshow("s", imgfg)
                        # cv2.waitKey(0)
                    else:
                        imgfg = img2fuben[0:heightimg2, ilistcs[j]:ilistce[j]]  # 3通道
                        imglist.append(imgfg)
                        # cv2.imshow("s2", imgfg)
                        # cv2.waitKey(0)



            else:
                imglist.append(img2)
                # cv2.imshow("s",img2)
                # cv2.waitKey(0)
        return imglist



def getdivide(imgpath):
    img=cv2.imread(imgpath,0)
    imglist = pinyinfenge(img)

    return imglist

if __name__ == '__main__':
    path='C:\\Users\MarkXu\\Desktop\\pinyintest/6F893EB13CD1E1E2A3CF389F7056A3E6.jpg'



    for i in getdivide(path):
        cv2.imshow("tem",i)
        cv2.waitKey(0)

    # for i in os.listdir('C:\\Users\\MarkXu\\Desktop\\dst\\chars/'):
    #
    #     dict1='C:\\Users\\MarkXu\\Desktop\\dst\\chars/'+i
    #     for j in os.listdir(dict1):
    #         img = cv2.imread(dict1+'/'+j)
    #         cv2.imshow("img",img)
    #         cv2.waitKey(100)
    #         imglist=pingyinfengge(img)#参数为你想裁剪的图片
    #         count=0
    #         for j in imglist:#显示试试
    #             cv2.imshow("temp",j)
    #             cv2.waitKey(0)
    #
    #             count+=1
