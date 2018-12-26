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

class Training_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, reshaped_size = (150, 150)):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.reshaped_size = reshaped_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
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

    return label_img, labels_res, unlabel_img
    

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
    X_train, y_label, unlabelled = load_train_data()

    X_train, y_label = reweight_sample(X_train, y_label)

    positive_ratio = np.sum(y_label, axis = 0) / len(y_label)
    print(positive_ratio)
    pass