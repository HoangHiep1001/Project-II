import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib
import os
import random

homePath = 'E:/AI/Python code/xuli/image/data/'
IMG_SIZE = 28
NUMBER_CLASS = 10
chars = [chr(i) for i in range(47,56)]
NUM_TRAIN = 6500
NUM_VAL = 7300
# ham xu li du lieu
def preprocessdata():
    data=[]
    label=[]
    for i,char in enumerate(chars):
        imgPath = homePath + char + '/'
        imgFiles = os.listdir(imgPath)
        for image in imgFiles:
            img = cv2.imread(imgPath+ image,0)
            data.append(img)
            label.append(i)
    l = len(label)
    shutfle = list(range(l))
    random.shuffle(shutfle)
    train_data = np.array(data)
    train_label = np.array(label)
    train_data = train_data[shutfle]
    train_label = train_label[shutfle]
    return data,label

data,label = preprocessdata()

train_x = data[:NUM_TRAIN]
train_y = label[:NUM_TRAIN]

valid_x = data[NUM_TRAIN:NUM_VAL]
valid_y = label[NUM_TRAIN:NUM_VAL]

test_x = data[NUM_VAL:]
test_y = data[NUM_VAL:]
np.save(homePath + '/train_x',train_x)
np.save(homePath + '/train_y',train_y)
np.save(homePath + '/valid_x',valid_x)
np.save(homePath + '/valid_y',valid_y)
np.save(homePath + '/test_x',test_x)
np.save(homePath + '/test_y',test_y)