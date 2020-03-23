# Data loading and reprocessing
import numpy as np
from numpy import csv
import pandas as pd
import tensorflow as tf
import os
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def readucr(filename): # Load data
    data = np.loadtxt(filename, dtype=str, delimiter = ',')
    data = data[:, 0: -1]  # 去掉最后一个label
    truncate = np.int(np.floor((len(data[0]) - 1) / 8) * 8)
    x = np.array(data[:, 0: truncate], dtype=np.float32)
    y = np.array(data[:, 1: (truncate + 1)], dtype=np.float32)
    x = x.reshape((x.shape[0], x.shape[1],1))
    y = y.reshape((y.shape[0], y.shape[1],1))
    print(f'{filename}: {x.shape}')
    return x, y

def readyahoo(filename):
    with open(filename, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        values = [row[1] for row in reader]
    data = np.array(values[1:], dtype=np.float32).reshape(1, len(values[1:]))
    truncate = np.int(np.floor((len(data[0]) - 1) / 8) * 8)
    x = np.array(data[:, 0: truncate], dtype=np.float32)
    y = np.array(data[:, 1: (truncate + 1)], dtype=np.float32)
    x = x.reshape((x.shape[0], x.shape[1],1))
    y = y.reshape((y.shape[0], y.shape[1],1))
    print(f'{filename}: {x.shape}')
    return x, y

def Data_MinMax_Scaler(x_train): # Normoaliazation
    sample_num = x_train.shape[0]
    for i in range(0, sample_num):
        scaler = MinMaxScaler(feature_range=(-1, 1)) # 归一化到 [-1, 1] 区间内
        x_train[i] = scaler.fit_transform(x_train[i]) # fit 获得最大值和最小值，transform 执行归一化
    return x_train

def is_abnormal(cnnfilepath, X_test, X_train, Y_train):
    mini_batch_size = 6

    model = load_model(cnnfilepath)

    threshold = model.evaluate(X_train, Y_train, batch_size=mini_batch_size, verbose=0)
    print('threshold is: {}'.format(threshold[0]))

    [loss, _] = model.evaluate(X_test, X_test, verbose=0)
    print('test loss is: {}'.format(loss))
    print('——————————————————————————————————————')

    if loss > (threshold[0] * 1.5):
        print('This is an abnormal data')
    else:
        print('This is a normal data')