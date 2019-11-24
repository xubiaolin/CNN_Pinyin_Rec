from flask import Flask
from flask import request
import uuid
import divider
import cv2
import train_and_val as pre

app = Flask(__name__)
filePath = 'uploadFile/'


@app.route('/index')
def index():
    return 'hello world'


@app.route('/judge/word', methods=['POST'])
def getpic():
    uploadFile = request.files['img']
    fileName = str(uploadFile.filename)
    print('单字的名称:' + fileName)
    uuidstr = str(uuid.uuid4()).replace('-', '')
    fileName = uuidstr + fileName[fileName.find('.'):]
    dstPath = filePath + fileName
    print(dstPath)
    uploadFile.save(dstPath)

    imglist=divider.getdivide(dstPath)
    for i in range(imglist):
        cv2.imshow("temp",i)
        cv2.moveWindow("temp",200,200)
        cv2.waitKey(1000)

    return '1'


@app.route('/judge',methods=['POST'])
def judgePinyin():
    uploadFile = request.files['file']
    fileName = str(uploadFile.filename)
    print('单字的名称:' + fileName)
    uuidstr = str(uuid.uuid4()).replace('-', '')
    fileName = uuidstr + fileName[fileName.find('.'):]
    dstPath = filePath + fileName
    print(dstPath)
    uploadFile.save(dstPath)

    # 开始分割
    imglist=divider.getdivide(dstPath)
    for i in range(len(imglist)):
        cv2.imwrite('cuttedPinYin/'+str(i)+".jpg",imglist[i])

    Ans_List=pre.predictForServer(imglist)
    #pre.predict('data/test/30_17.jpg', False)
    Ans_List=''.join(Ans_List)
    print(Ans_List)

    return Ans_List


if __name__ == '__main__':
    app.run()
