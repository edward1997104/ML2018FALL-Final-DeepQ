#!/usr/bin/env python

import pandas as pd
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
import cv2
from keras.utils import Sequence
from keras.preprocessing.image import *
import numpy as np
import copy 
from aug import random_rotation
import keras.applications

class Training_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, reshaped_size = (150, 150), epsilon = 0.25):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.reshaped_size = reshaped_size
        self.epsilon = epsilon

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x, batch_y = self.__data_generation(idx)

        if np.random.rand() < self.epsilon:
            batch_x = self.__augmentation(batch_x)

        return batch_x, batch_y

    def __augmentation(self, batch_x, epsilon = 0.25):
        shape = batch_x.shape
        batch_x = np.array([random_rotation(x) for x in batch_x]).reshape(shape)

        batch_x = batch_x.reshape(shape)

        return batch_x

    def __data_generation(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            cv2.resize(cv2.imread(file_name), dsize = self.reshaped_size)
               for file_name in batch_x]), np.array(batch_y)

class Testing_Generator(Sequence):
    
    def __init__(self, image_filenames, batch_size, reshaped_size = (150, 150)):
        self.image_filenames= image_filenames
        self.batch_size = batch_size
        self.reshaped_size = reshaped_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            cv2.resize(cv2.imread(file_name), dsize = self.reshaped_size)
               for file_name in batch_x])

class Unsupervised_Generator(Sequence):
    
    def __init__(self, image_filenames, batch_size, reshaped_size = (150, 150)):
        self.image_filenames= image_filenames
        self.batch_size = batch_size
        self.reshaped_size = reshaped_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.array([cv2.resize(cv2.imread(file_name), dsize = self.reshaped_size) for file_name in batch_x])
        return X, keras.applications.inception_v3.preprocess_input(X)

def load_train_data(train_img_folder = './data/data/images/', path = './data/ntu_final_2018/train.csv'):
    
    # loading the dataset
    train_df = pd.read_csv(path, engine='python')
    train_df['Labels'].fillna(-1, inplace = True)

    img_idx, labels = train_df['Image Index'], train_df['Labels']

    label_img, labels_res, unlabel_img = [], [], []

    print("Loading Training Images and Labels")
    for img_id, label in zip(img_idx, labels):
        if label != -1:
            label_img.append(train_img_folder + img_id)
            label = [int(x) for x in label.split(" ")]
            labels_res.append(label)
        else:
            unlabel_img.append(train_img_folder + img_id)
    
    print("Done Training Images and Labels")                                                                      
    label_img, labels_res, unlabel_img = label_img, np.array(labels_res), unlabel_img
    
    
    labeled_img_idx, labeled_patient,  labeled_patient_id = \
        train_df['Image Index'][:10001], train_df['Labels'][:10001], train_df['Patient ID'][:10001]
    
    label_to_imgs = { i: [] for i in range(14)}
    img_flag = {}

    for i in range(10001):
        labels = labels_res[i].nonzero()[0]
        for label in labels:
            label_to_imgs[label].append(labeled_img_idx[i])
        img_flag[labeled_img_idx[i]] = [False, i]

    return label_img, labels_res, unlabel_img, label_to_imgs, img_flag
    

def reweight_sample(image_filenames, labels, per_class = 2000):

    np.random.seed(42)
    result_filenames, result_labels = [], []
    pos_sample = np.sum(labels, axis = 1)
    indicies = np.where(pos_sample == 0)[0]

    print(indicies)
    cnt = 0
    while cnt < int(per_class * 3):
        for i in indicies:
            result_filenames.append(image_filenames[i])
            result_labels.append(labels[i])
            cnt += 1
            if cnt >= per_class:
                break
    
    for j in range(labels.shape[1]):
        indicies = np.where(labels[:,j] > 0)[0]
        cnt = 0
        while cnt < per_class:
            for i in indicies:
                result_filenames.append(image_filenames[i])
                result_labels.append(labels[i])
                cnt += 1
                if cnt >= per_class:
                    break 
    
    # p = np.random.permutation(result_filenames)
    return result_filenames, np.array(result_labels)

def split_dataset(y_label, label_to_imgs, img_flag, test_ration = 0.15):
    distribution = dist(y_label)
    test_size = list(map(lambda x: int(x*test_ration), distribution))
    retrieve_seq = np.argsort(test_size)
    print(test_size)
    print(retrieve_seq)
    test_ids = []
    for label in retrieve_seq:
        for i in range(test_size[label]):
            img_name, flag = get_img_flag(label, label_to_imgs, img_flag)
            if img_name == None:
                print(len(test_ids), label)
                break
            img_flag[img_name][0] = True 
            test_ids.append(flag[1])
            
    train_ids = []
    for flag in img_flag:
        if img_flag[flag][0] == False:
            train_ids.append(img_flag[flag][1])
    
    return train_ids, test_ids
            
    
def get_img_flag(target_label, label_to_imgs, img_flag):
    for i in label_to_imgs[target_label]:
        if img_flag[i][0] == False:
            return i, img_flag[i]
    return None, None

def dist(y_label):
    stat = [list(y.nonzero()[0])  for y in y_label]
    tmp = list(filter(lambda x: x!= [], stat))
    freq = [0]*14
    for i in tmp:
        for index in i:
            freq[index] += 1
    
    return freq

def load_test_idxs(path = './data/ntu_final_2018/test.csv', test_img_folder = './data/data/images/'):
    
    test_df = pd.read_csv(path, engine='python')
    img_idx = test_df['Image Index']
    output_idx = copy.deepcopy(img_idx)
    img_idx = [ test_img_folder + img_id for img_id in img_idx]

    return img_idx, output_idx

def generate_batch(x, y, batch_size):
    pass

def show_img(arr):
    plt.imshow(arr, cmap=plt.get_cmap('gray'))

if __name__ == '__main__':
    X_train, y_label, unlabelled, label_to_imgs, img_flag = load_train_data()
    train_ids, test_ids = split_dataset(y_label, label_to_imgs, img_flag, test_ration=0.20)

#    X_train, y_label = reweight_sample(X_train, y_label)

#    positive_ratio = np.sum(y_label, axis = 0) / len(y_label)
#    print(positive_ratio)
    pass