from flask import Flask
from flask import request
import uuid
import divider
import time
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
    # print('单字的名称:' + fileName)
    uuidstr = str(uuid.uuid4()).replace('-', '')
    fileName = uuidstr + fileName[fileName.find('.'):]
    dstPath = filePath + fileName
    # print(dstPath)
    uploadFile.save(dstPath)

    imglist = divider.getdivide(dstPath)
    for i in range(imglist):
        cv2.imshow("temp", i)
        cv2.moveWindow("temp", 200, 200)
        cv2.waitKey(1000)

    return '1'


@app.route('/judge/pinyin', methods=['POST'])
def judgePinyin():
    try:
        uploadFile = request.files['file']
        right_ans = request.form.get('right_ans')
        print("right_ans:" + right_ans)
        fileName = str(uploadFile.filename)
        # print('单字的名称:' + fileName)
        uuidstr = str(uuid.uuid4()).replace('-', '')
        fileName = uuidstr + fileName[fileName.find('.'):]
        dstPath = filePath + fileName
        print(dstPath)
        uploadFile.save(dstPath)

        # 开始分割
        imglist = divider.getdivide(dstPath)
        for i in range(len(imglist)):
            cv2.imwrite('cuttedPinYin/' + str(i) +'_'+str(time.time())+ ".jpg", pre.dealwithimg(imglist[i]))

        Ans_List = pre.predictForServer(imglist)
        Ans_List = ''.join(Ans_List)
        print(Ans_List)
        print('-------------------------------')
        print()
        if Ans_List == right_ans:
            return '1'
        return '0'
    except:
        return '2'


if __name__ == '__main__':
    app.run(port=5001)
