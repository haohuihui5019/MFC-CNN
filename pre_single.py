from keras.utils import to_categorical
import numpy as np
import os
# This file is used to build input tags and datasets
def get_files(file_dir,savedir):
    image_list = []
    label_list = []
    name_dic = {}
    name_count=0
    # Load data path and write tag
    for file in os.listdir(file_dir):
        name = str(file)
        name_dic[name] = name_count    # The number 1 2 3 ……indicate each class
        name_count+=1
        file_count = 0
        for key in os.listdir(file_dir + file):
            file_count+=1
            image_list.append(file_dir + '\\' + file + '\\' + key)
            label_list.append(name_dic[file])

    image_list = np.hstack(image_list)
    label_list = np.hstack(label_list)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # train_img, test_img = train_test_split(temp, train_size=0.8)
    # vali_img,  test_img = train_test_split(test_img, train_size=0.5)

    # train_image_list = list(train_img[:, 0])
    image_list = list(temp[:, 0])
    # vali_image_list = list(vali_img[:, 0])

    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    label_list1 = to_categorical(label_list, num_classes=CLASS_NUM)

    np.save(savedir+'test_image_list1', image_list)

    np.save(savedir+'test_label_list', label_list1)

    return  image_list,label_list1,np.array(label_list1)

if __name__=='__main__':
    train_file_path = r"F:\multi_inputs_data\test_image/"  #
    save__path='F:\multi_inputs_data\dataset_aug_2/'
    CLASS_NUM = 4 # Number of categories
    # norm_size = 224
    testX,testY,test_label_list = get_files(train_file_path,save__path) # 导入数据集
    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    #                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    #                          horizontal_flip=True, fill_mode="nearest")
    # train(aug, trainX, trainY, testX, testY,test_label_list)
