import os
from keras.preprocessing.image import img_to_array
import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D,\
    AveragePooling2D, Concatenate
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
# from generate_valdata import Generate_valData
# from keras.utils import multi_gpu_model
import datetime
import numpy as np
import cv2

def trans_size(np_img,img_size):
    train_img_trans = []
    for i in range(len(np_img)):
        train_img_trans.append(cv2.resize(np_img[i],(img_size,img_size),interpolation=cv2.INTER_CUBIC))
    train_img_trans=np.array(train_img_trans)
    train_img_trans = np.array(train_img_trans, dtype="float") / 255.0
    return  train_img_trans
def input1(np_img,img_size):
    train_img_trans = []
    for i in range(len(np_img)):
        img = cv2.imread(np_img[i])
        w,h,_ = img.shape
        # padding operation
        if w > h:
            default = cv2.copyMakeBorder(img, 0, 0, int((w - h) / 2), int((w - h) / 2), cv2.BORDER_CONSTANT, value=[255,255,255])
        else:
            default = cv2.copyMakeBorder(img, int((h - w) / 2), int((h - w) / 2), 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
        train_img_trans.append(img_to_array(cv2.resize(default,(img_size,img_size),interpolation=cv2.INTER_CUBIC)))
    train_img_trans=np.array(train_img_trans)
    train_img_trans = np.array(train_img_trans, dtype="float") / 255.0
    return  train_img_trans
def input2(np_img,img_size):
    train_img_trans = []
    for i in range(len(np_img)):
        original = cv2.imread(np_img[i])
        groundtruth = original[:, :, 0]
        blurred = cv2.GaussianBlur(groundtruth, (5, 5), 0)  # Gaussian smoothing using a 5 x 5 kernel
        thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)[1]  # thresholding
        thresh = 255 - thresh
        contours, cnt = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        image1 = original[int(y):int(y + h), int(x):int(x + w)]
        # Target size after segmentation
        w, h, _ = image1.shape
        if w > 100 and h > 100:
            # padding
            if w > h:
                default = cv2.copyMakeBorder(image1, 0, 0, int((w - h) / 2), int((w - h) / 2), cv2.BORDER_CONSTANT, value=[255,255,255])
            else:
                default = cv2.copyMakeBorder(image1, int((h - w) / 2), int((h - w) / 2), 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
        else:
            # For image that fail in target detection, crop directly from the original image
            w_or = original.shape[0]
            h_or = original.shape[1]
            default = original[int(0.2 * h_or):int(0.8 * h_or), int(0.2 * w_or):int(0.8 * w_or)]
           # padding
            w1, h1,_ = default.shape
            if w1 > h1:
                default = cv2.copyMakeBorder(default, 0, 0, int((w1 - h1) / 2), int((w1 - h1) / 2), cv2.BORDER_CONSTANT, value=[255,255,255])
            else:
                default = cv2.copyMakeBorder(default, int((h1 - w1) / 2), int((h1 - w1) / 2), 0, 0,cv2.BORDER_CONSTANT, value=[255,255,255])  # 延长边界值填充用cv2.BORDER_REPLICATE
        train_img_trans.append(img_to_array(cv2.resize(default, (img_size, img_size), interpolation=cv2.INTER_CUBIC)))
    train_img_trans = np.array(train_img_trans)
    train_img_trans = np.array(train_img_trans, dtype="float") / 255.0
    return train_img_trans
def input34(np_img,img_size):
    train_img_trans = []
    for i in range(len(np_img)):
        img = np_img[i].copy()
        w, h, _ = img.shape
        image1 = img[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]
        image1 = cv2.resize(image1,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
        train_img_trans.append(img_to_array(image1))
    train_img_trans=np.array(train_img_trans)
    train_img_trans = np.array(train_img_trans, dtype="float") / 255.0

    return train_img_trans
def getdata(datadir):
    train_img = np.load(datadir+'train_image_list1.csv.npy')
    train_img112 = input2(train_img,112)
    train_img224 = input1(train_img, 224)
    train_img56 = input34(train_img112,56)
    train_img28 = input34(train_img56,28)
    np.save(savedir + 'train_image224.npy', train_img224)
    np.save(savedir + 'train_image112.npy', train_img112)
    np.save(savedir + 'train_image56.npy', train_img56)
    np.save(savedir + 'train_image28.npy', train_img28)

    traindata = [train_img224, train_img112, train_img56, train_img28]

    vali_img = np.load(datadir+'vali_image_list1.csv.npy')
    vali_img112 = input2(vali_img,112)
    vali_img224 = input1(vali_img,224)
    vali_img56 = input34(vali_img112,56)
    vali_img28 = input34(vali_img56,28)
    np.save(savedir + 'vali_image224.npy', vali_img224)
    np.save(savedir + 'vali_image112.npy', vali_img112)
    np.save(savedir + 'vali_image56.npy', vali_img56)
    np.save(savedir + 'vali_image28.npy', vali_img28)


    validata= [vali_img224, vali_img112, vali_img56, vali_img28]

    test_img = np.load(datadir+'test_image_list1.csv.npy')
    test_img112 = input2(test_img, 112)
    test_img224 = input1(test_img, 224)
    test_img56 = input34(test_img112, 56)
    test_img28 = input34(test_img56, 28)
    np.save(savedir+'test_image224.npy',test_img224)
    np.save(savedir + 'test_image112.npy', test_img112)
    np.save(savedir + 'test_image56.npy', test_img56)
    np.save(savedir + 'test_image28.npy', test_img28)


    testdata = [test_img224, test_img112, test_img56, test_img28]
    return traindata, validata, testdata,
# Execute this file to process the original datasets into 4 scale inputs
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    datadir = r"H:\haoxia\images\my_process\dataset/"
    savedir = r'H:\haoxia\images\my_process\dataset/'
    traindata,  validata, testdata,  = getdata(datadir)
