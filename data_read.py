import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import shutil

def Data_read(raw_data_dir_, data_dir_):

    data_dir = data_dir_
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_data_dir = raw_data_dir_
    deepest_train_txt = "{}/deepest_train.txt".format(raw_data_dir)
    deepest_test_txt = "{}/deepest_test.txt".format(raw_data_dir)
    deepest_train_list = []
    deepest_test_list = []

    with open(deepest_train_txt, 'r') as f:
        lines = f.read().splitlines()
        deepest_train_list = lines

    with open(deepest_test_txt, 'r') as f:
        lines = f.read().splitlines()
        deepest_test_list = lines

    deepest_train_label = []
    for i in range(len(deepest_train_list)):
        deepest_train_label.append(deepest_train_list[i].split('/')[2])

    deepest_test_label = []
    for i in range(len(deepest_test_list)):
        deepest_test_label.append(deepest_test_list[i].split('/')[2])

    dir_save_train = os.path.join(data_dir,'train')
    dir_save_val = os.path.join(data_dir,'val')
    dir_save_test = os.path.join(data_dir,'test')

    if not os.path.exists(dir_save_train):
        os.makedirs(dir_save_train)
    if not os.path.exists(dir_save_val):
        os.makedirs(dir_save_val)
    if not os.path.exists(dir_save_test):
        os.makedirs(dir_save_test)

    for i in range(len(deepest_train_list)):
        # 비슷한 label끼리 뭉쳐있는듯 보여 5번째 이미지를 val로 불러옴
        if i%5 != 0:
            train_label_folder = os.path.join(dir_save_train,deepest_train_label[i])
            train_image_name = deepest_train_list[i].split('/')[-1]
            if not os.path.exists(train_label_folder):
                os.makedirs(train_label_folder)
            shutil.copy(os.path.join(raw_data_dir,deepest_train_list[i]),os.path.join(train_label_folder,train_image_name))

        else:
            val_label_folder = os.path.join(dir_save_val, deepest_train_label[i])
            val_image_name = deepest_train_list[i].split('/')[-1]
            if not os.path.exists(val_label_folder):
                os.makedirs(val_label_folder)
            shutil.copy(os.path.join(raw_data_dir, deepest_train_list[i]),
                        os.path.join(val_label_folder, val_image_name))

    for i in range(len(deepest_test_list)):
        test_label_folder = os.path.join(dir_save_test, deepest_test_label[i])
        test_image_name = deepest_test_list[i].split('/')[-1]
        if not os.path.exists(test_label_folder):
            os.makedirs(test_label_folder)
        shutil.copy(os.path.join(raw_data_dir, deepest_test_list[i]), os.path.join(test_label_folder, test_image_name))




