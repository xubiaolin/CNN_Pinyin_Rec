import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import random
import os
import json
import time
from MakeData import cut
from MakeData import resize
EPOCH = 120
BATCH_SIZE = 100
IMAGE_PATH = "Data/train/"

IMAGE_MUMBER = len(os.listdir(IMAGE_PATH))

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHAR_SET_LEN = 50  # 数字大小,后期要修改
xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
ys = tf.placeholder(tf.float32, [None, CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # 防止过拟合
x_image = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])


# 计算weight
def weigth_variable(shape):
    # stddev : 正态分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布
    return tf.Variable(initial)


# 计算biases
def bias_varibale(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 计算卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 训练网络
def code_cnn():
    # 第一个卷积层
    W_conv1 = weigth_variable([5, 5, 1, 32])
    b_conv1 = weigth_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # 14*14*32
    # 第二个卷积层
    W_conv2 = weigth_variable([5, 5, 32, 64])
    b_conv2 = weigth_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # 7*7*64
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)
    # 三层全连接层
    W_fc1 = weigth_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_varibale([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过度拟合
    # 第四层全连接层
    W_fc2 = weigth_variable([1024, CHAR_SET_LEN])
    b_fc2 = bias_varibale([CHAR_SET_LEN])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return prediction


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 文本转向量
# def text2vec(text):
#     text_len = len(text)
#     vector = np.zeros(1 * CHAR_SET_LEN)
#
#     def char2pos(c):
#         if c == '_':
#             k = 62
#             return k
#         k = ord(c) - 48
#         if k > 9:
#             k = ord(c) - 55
#             if k > 35:
#                 k = ord(c) - 61
#                 if k > 61:
#                     raise ValueError('No Map')
#         return k
#
#     for i, c in enumerate(text):
#         idx = i * CHAR_SET_LEN + char2pos(c)
#         vector[idx] = 1
#     return vector


# 生成一个训练batch

# 文本转向量
def text2vec(text):
    # print("输入的text:{}".format(text))
    text_len = len(text)
    vector = np.zeros(1 * CHAR_SET_LEN)
    vector[int(text) - 1] = 1
    return vector


def get_next_batch(batch_size, each, images, labels):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, CHAR_SET_LEN])

    def get_text_and_image(i, each):
        image_num = each * batch_size + i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image

    for i in range(batch_size):
        text, image = get_text_and_image(i, each)
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


# 随机生成一个训练batch
def get_random_batch(batch_size, images, labels):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, 1 * CHAR_SET_LEN])

    def get_captcha_text_and_image(i):
        image_num = i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image

    for i in range(batch_size):
        text, image = get_captcha_text_and_image(random.randint(0, len(images) - 1))
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


# 计算准确率
# batch_x_test, batch_y_test = get_random_batch(BATCH_SIZE, test_image_paths, test_labels)
def compute_accuracy(v_xs, v_ys, sess):  # 传入测试样本和对应的label
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


prediction = code_cnn()
# 用于保存和载入模型
saver = tf.train.Saver()


def train_code_cnn(image_paths, labels, flag='train'):
    # 定义网络
    global prediction
    # 计算loss cross_entropy
    with tf.Session() as sess:
        if flag == 'train':
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            # 初始化variable
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(EPOCH):
                # 每个epoch
                for each in range(int(IMAGE_MUMBER / BATCH_SIZE)):
                    batch_x, batch_y = get_next_batch(BATCH_SIZE, each, image_paths, labels)
                    _, loss_ = sess.run([train_step, cross_entropy]
                                        , feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
                    print("epoch: %d  iter: %d/%d   loss: %f"
                          % (epoch + 1, BATCH_SIZE * each, IMAGE_MUMBER, loss_))
                # 训练样本测试准确率
                test_iamge_path = "Data/test/"
                test_image_paths, test_labels = get_image_path_labels(test_iamge_path)

                batch_x_test, batch_y_test = get_random_batch(BATCH_SIZE, test_image_paths, test_labels)

                accuracy_test = compute_accuracy(batch_x_test, batch_y_test, sess)
                print("测试样本测试 epoch: %d  acc: %f" % (epoch + 1, accuracy_test))

                batch_x_test, batch_y_test = get_random_batch(BATCH_SIZE, image_paths, labels)
                accuracy = compute_accuracy(batch_x_test, batch_y_test, sess)
                print("训练样本测试 epoch: %d  acc: %f" % (epoch + 1, accuracy))

            saver.save(sess, './model/image_model')
            print("模型保存成功")
        else:
            saver.restore(sess, './model/image_model')
            test_iamge_path = "Data/test/"
            test_image_paths, test_labels = get_image_path_labels(test_iamge_path)

            batch_x_test, batch_y_test = get_random_batch(BATCH_SIZE, test_image_paths, test_labels)

            accuracy_test = compute_accuracy(batch_x_test, batch_y_test, sess)
            print("测试样本测试 epoch: %d  acc: %f" % (1, accuracy_test))


# 获取预测值
def getSingleImageBatch(image_path):
    batch_x = np.zeros([1, IMAGE_HEIGHT * IMAGE_WIDTH])
    captcha_image = cv2.imread(image_path, 0)
    captcha_image=dealwithimg(captcha_image)
    # kernel = np.ones((3, 3), np.uint8)
    # dilation = cv2.dilate(captcha_image, kernel)  # 膨胀
    captcha_image = cv2.resize(captcha_image, (28, 28))
    captcha_image = np.array(captcha_image)
    image = convert2gray(captcha_image)
    batch_x[0, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
    return batch_x






def predict(image_path, isList=False):
    global prediction
    with tf.Session() as sess:
        saver.restore(sess, './model/image_model')
        if isList:
            count = 0
            for i in os.listdir(image_path):
                v_xs = getSingleImageBatch(image_path + i)
                y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
                pre = y_pre[0]

                index = np.where(y_pre == np.max(y_pre))
                index = index[1][0] + 1  # 概率最大的下标
                # print(index)
                pre.sort()
                # print(pre[-1], pre[-2])
                itemindex = np.argwhere(y_pre == pre[-2])
                itemindex = itemindex[0][1] + 1

                with open('num_char.json', 'r') as f:
                    dict = json.loads(f.read())


                print('识别图片:{}'.format(image_path + i))
                print('预测的结果为:' + dict[str(index)], '概率为:{}'.format(np.max(y_pre)))
                print('预测的结果为:' + dict[str(itemindex)], '概率为:' + str(pre[-2]))
                print()
        else:
            v_xs = getSingleImageBatch(image_path)
            y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
            pre = y_pre[0]

            index = np.where(y_pre == np.max(y_pre))
            index = index[1][0] + 1  # 概率最大的下标
            print(index)
            pre.sort()
            print(pre[-1], pre[-2])
            itemindex = np.argwhere(y_pre == pre[-2])
            itemindex = itemindex[0][1] + 1

            with open('num_char.json', 'r') as f:
                dict = json.loads(f.read())
            print('预测的结果为:' + dict[str(index)], '概率为:{}'.format(np.max(y_pre)))
            print('预测的结果为:' + dict[str(itemindex)],'概率为:'+str(pre[-2]))


def predictForServer(imglist):
    global prediction

    def getSingleImageBatch(image):
        batch_x = np.zeros([1, IMAGE_HEIGHT * IMAGE_WIDTH])
        captcha_image = image
        captcha_image = dealwithimg(captcha_image)

        captcha_image = cv2.resize(captcha_image, (28, 28))

        captcha_image = np.array(captcha_image)
        image = convert2gray(captcha_image)
        batch_x[0, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        return batch_x

    with tf.Session() as sess:
        saver.restore(sess, './model/image_model')
        count = 0
        list=[]
        for i in imglist:
            v_xs=getSingleImageBatch(i)
            y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
            pre = y_pre[0]

            index = np.where(y_pre == np.max(y_pre))
            index = index[1][0] + 1  # 概率最大的下标
            # print(index)
            pre.sort()
            # print(pre[-1], pre[-2])
            itemindex = np.argwhere(y_pre == pre[-2])
            itemindex = itemindex[0][1] + 1

            with open('num_char.json', 'r') as f:
                dict = json.loads(f.read())

            list.append(dict[str(index)])

        return list

# 根据路径得到文本的内容
def getStrContent(path):
    return open(path, 'r', encoding="utf-8").read()


# 返回 训练样本路径的list 和 对应的标签用来以后训练
def get_image_path_labels(IMAGE_PATH=IMAGE_PATH):
    image_path = IMAGE_PATH
    image_paths = []
    labels = []

    for i in os.listdir(image_path):
        image_paths.append(image_path + i)
        labels.append(i.split('_')[0])

    return image_paths, labels


def main():
    # 得到训练样本路径list和标签的list

    image_paths, labels = get_image_path_labels()
    train_code_cnn(image_paths, labels, 'train')


def dealwithimg(img):
    width,height=img.shape
    if width==height:
        return img
    x1, x2, y1, y2 = cut(img)

    # print(x1, x2, y1, y2)
    temp = img[y1:y2, x1:x2]
    temp=resize(temp)
    return temp

if __name__ == '__main__':
    # predict('Data/test/',True)
    main()
    #predict('data/test/30_17.jpg', False)
    #predict('cuttedPinYin/1.jpg',False)
    path='C:\\Users\\MarkXu\\Desktop\\wrong\\'
    predict(path,True)

