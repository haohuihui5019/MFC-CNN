import os

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
# from keras.utils import multi_gpu_model
import datetime
import numpy as np
import cv2

def scheduler(epoch):
    learning_rate_init = 0.005
    if epoch > 20:
        learning_rate_init = 0.0005
    if epoch > 40:
        learning_rate_init = 0.0001
    return learning_rate_init

# load dataset
def getdata(datadir):
    train_label = np.load(datadir+'train_label_list.csv.npy')
    vali_label = np.load(datadir+'vali_label_list.csv.npy')
    test_label = np.load(datadir+'test_label_list.csv.npy')

    train_img224 = np.load(datadir+'train_image224.npy')
    train_img112 = np.load(datadir + 'train_image112.npy')
    train_img56 = np.load(datadir + 'train_image56.npy')
    train_img28 = np.load(datadir + 'train_image28.npy')
    traindata = [train_img224, train_img112, train_img56, train_img28]

    vali_label = np.load(datadir+'vali_label_list.csv.npy')
    vali_img224 = np.load(datadir + 'vali_image224.npy')
    vali_img112 = np.load(datadir + 'vali_image112.npy')
    vali_img56 = np.load(datadir + 'vali_image56.npy')
    vali_img28 = np.load(datadir + 'vali_image28.npy')
    validata= [vali_img224, vali_img112, vali_img56, vali_img28]

    test_img224 = np.load(datadir + 'test_image224.npy')
    test_img112 = np.load(datadir + 'test_image112.npy')
    test_img56 = np.load(datadir + 'test_image56.npy')
    test_img28 = np.load(datadir + 'test_image28.npy')
    testdata = [test_img224, test_img112, test_img56, test_img28]
    return traindata, train_label, validata, vali_label, testdata, test_label

def multi_inputs():
    input1 = Input(shape=(224, 224, 3), name='input1')
    # Block 1, 2
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv1', input_shape=(224, 224, 3))(
        input1)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)

    input2 = Input(shape=(112, 112, 3), name='input2')
    x2 = Conv2D(64, (3, 3), activation='elu', padding='same', name='block1_conv3')(input2)
    x2 = BatchNormalization()(x2)
    # cascade
    x = Concatenate(axis=-1, name='concatenate2')([x, x2])
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='block2_conv4')(x)
    x = BatchNormalization()(x)

    input3 = Input(shape=(56, 56, 3), name='input3')
    x3 = Conv2D(128, (3, 3), activation='elu', padding='same', name='block2_conv5')(input3)
    x3 = BatchNormalization()(x3)

    x = Concatenate(axis=-1, name='concatenate3')([x, x3])
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(156, (3, 3), activation='elu', padding='same', name='block2_conv6')(x)
    x = BatchNormalization()(x)

    input4 = Input(shape=(28, 28, 3), name='input4')
    x4 = Conv2D(156, (3, 3), activation='elu', padding='same', name='block2_conv7')(input4)
    x4 = BatchNormalization()(x4)

    x = Concatenate(axis=-1, name='concatenate4')([x, x4])
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(250, (3, 3), activation='elu', padding='same', name='block2_conv8')(x)
    x = BatchNormalization()(x)

    x = AveragePooling2D((4, 4), strides=4)(x)
    # Classification block,
    x = Flatten()(x)
    # x = Dense(2048, activation='relu', name='fc1')(x)
    # x = Dropout(0.5)(x)
    x = Dense(2048, activation='elu', name='fc2')(x)
    # x = Dropout(0.5)(x)

    x = Dense(4, activation='softmax')(x)
    model = Model(inputs=[input1, input2, input3, input4], outputs=x)
    model.summary()
    return model
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    EPOCH = 60
    BATCH_SIZE = 20
    datadir= r'F:\multi_inputs_data\dataset_aug/'

    logdir ='./logs'

    traindata, train_label, validata, vali_label, testdata, test_label = getdata(datadir)
    train_num=int(len(train_label))
    aug = ImageDataGenerator()
    model = multi_inputs()
    plot_model(model, to_file='multi-input.png', show_shapes='True')
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    lr_rate = LearningRateScheduler(scheduler)
    tb_cb = TensorBoard(log_dir=logdir, histogram_freq=0)
    cbks=[lr_rate, tb_cb]
    starttime = datetime.datetime.now()
    history=model.fit(traindata,
                      train_label,
                      batch_size=BATCH_SIZE ,
                      epochs=EPOCH,
                      validation_data=[validata, vali_label],
                      shuffle=True,
                      callbacks=cbks,
                      verbose=1
                      )
    endtime = datetime.datetime.now()
    print ('training time：',endtime - starttime)
    starttime = datetime.datetime.now()
    score = model.evaluate(testdata, test_label, verbose=0)  # evaluate the trained model
    endtime = datetime.datetime.now()
    print ('test time：',endtime - starttime)
    print('score is: ', score)

    # np.savetxt('./results/multiPatches_train_loss.txt', history.history["loss"], fmt="%f", delimiter="\n")
    # np.savetxt('./results/multiPatches_train_acc.txt', history.history["accuracy"], fmt="%f", delimiter="\n")
    # np.savetxt('./results/multiPatches_val_loss.txt', history.history["val_loss"], fmt="%f", delimiter="\n")
    # np.savetxt('./results/multiPatches_val_acc.txt', history.history["val_accuracy"], fmt="%f", delimiter="\n")
    # print (endtime - starttime)

    model.save('./model/multi_patches_model.h5')
    test_labels = [np.argmax(one_hot) for one_hot in test_label]
    print('test_labels:',test_labels)
    result = model.predict(testdata)

    predict = np.argmax(result, axis=1)

    confusion_mat = confusion_matrix(test_labels,predict)
    print('confusion matrix:',confusion_mat)
    print(classification_report(test_labels, predict))
    precision=metrics.precision_score(test_labels, predict, average='macro')
    recall=metrics.recall_score(test_labels, predict, average='macro')
    f1=metrics.f1_score(test_labels, predict, average='weighted')
    print('precision',precision,'recall:',recall,'f1:',f1)

    #=============output the predict result===============
    # predict_test = model.predict(testdata)
    # test_label2=[]
    # for i in range(len(test_labels)):
    #     test_label2.append([test_labels[i]])
    # print(test_label2)
    # print(predict_test)
    #
    # result=np.append(predict_test,test_label2,axis=1)
    #
    # print(result)
    # print(np.hstack((predict_test,test_labels)))
