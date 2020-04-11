#!/usr/bin/env python
#-*- coding: UTF-8 -*-
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def readUcrTsv(filename, timeWinNum, onlyGru=False, method=1):  # 交叉组合样本
    """
    @filename: 文件路径名
    @timeWinNum: 时间窗个数
    @onlyGru: False为考虑CNN对输入样本的要求，即从gru出来的timesteps为 8 的倍数，sample数为 6 的倍数
    @method: method=1 表示将样本按sensor类别交叉分组后，将多个sensor类别合成到最后一个维度
             method=2 表示将样本按sensor类别直接分组，用于1个sensor对应一个multi_Gru，多个sensor独立训练多个gru
    """
    # 读取源文件
    df = pd.read_csv(filename, delimiter='\t', header=None)
    df.rename(columns={0: 'class'}, inplace=True)
    sensors_list = df['class'].unique()
    sensors_num = sensors_list.max()

    # 将原数据按照类别分组 并用字典存储被一类数据
    d = dict()
    for clas in sensors_list:
        temp_data = df[df['class'].isin([clas])]
        d[f'class_' + str(clas)] = temp_data
        d[f'class_' + str(clas)].reset_index(drop=True, inplace=True)

    if(method==1):
        # 找出所有类中样本最少的一类的样本数 以此为拼接上限
        min_rows_num = d['class_1'].shape[0]
        for i in range(1, sensors_num + 1):
            if (d['class_' + str(i)].shape[0] <= min_rows_num):
                min_rows_num = d['class_' + str(i)].shape[0]

        # 交叉组合生成新的DataFrame
        old_data = pd.DataFrame()
        for i in range(min_rows_num):
            for j in range(1, sensors_num + 1):
                old_data = old_data.append(d['class_' + str(j)].loc[i], ignore_index=True)

        # 删除class列
        old_data.drop(columns=['class'], inplace=True)

        # 转成numpy数组
        data = old_data.values
        # 生成训练集和测试集
        old_timesteps = data.shape[1]
        if (onlyGru == True):
            x = data[:, 0:-1]
            y = data[:, 1:]
            x = x.reshape(-1, sensors_num, old_timesteps).swapaxes(2, 1)
            y = y.reshape(-1, sensors_num, old_timesteps).swapaxes(2, 1)
            if (x.shape[0] % 6 != 0):
                a = x[-1].reshape(1, x[-1].shape[0], x[-1].shape[1])
                b = y[-1].reshape(1, y[-1].shape[0], y[-1].shape[1])
                imputation = 6 - (x.shape[0] % 6)
                for i in range(imputation):
                    x = np.append(x, a, axis=0)
                    y = np.append(y, b, axis=0)
            print("Data Only for GRU!")
            print(f'{filename}: {x.shape}')
            return x, y, sensors_num, sensors_num
        else:
            trunc = int(np.floor((old_timesteps - 1) / timeWinNum) * timeWinNum)
            x = data[:, 0:trunc]
            y = data[:, 1:(trunc + 1)]
            timesteps = x.shape[1]
            x = x.reshape(-1, sensors_num, timesteps).swapaxes(2, 1)
            y = y.reshape(-1, sensors_num, timesteps).swapaxes(2, 1)
            if (x.shape[0] % 6 != 0):
                a = x[-1].reshape(1, x[-1].shape[0], x[-1].shape[1])
                b = y[-1].reshape(1, y[-1].shape[0], y[-1].shape[1])
                imputation = 6 - (x.shape[0] % 6)
                for i in range(imputation):
                    x = np.append(x, a, axis=0)
                    y = np.append(y, b, axis=0)
            print("Data for GRU and CNN!")
            print(f'{filename}: {x.shape}')
            return x, y, sensors_num, int(timesteps / timeWinNum)
    else:
        ls = []
        max_sample_num = 1
        for clas in sensors_list:
            d[f'class_' + str(clas)] = d[f'class_' + str(clas)].drop(columns=['class']).copy(deep=True)
            if(d['class_' + str(clas)].shape[0] >= max_sample_num):
                max_sample_num = d['class_' + str(clas)].shape[0]
            ls.append(np.expand_dims(d[f'class_' + str(clas)].values, 2))  # （342，1639，1）
        num = int(np.ceil(max_sample_num / 6) * 6)
        ls.reverse()
        a = np.array(ls)
        trunc = int(np.floor((a[0].shape[1] - 1) / timeWinNum) * timeWinNum)
        x = a.copy()
        y = a.copy()
        for i in range(sensors_num):
            x[i] = x[i][:, 0:trunc]  # （342，1632，1）
            y[i] = y[i][:, 1:trunc+1]
        time = x[0].shape[1]
        for i in range(sensors_num):
            imputation = num - x[i].shape[0]
            if(imputation == 0):
                continue
            else:
                a = x[i][-1].reshape(1, x[i].shape[1], x[i].shape[2])
                b = y[i][-1].reshape(1, y[i].shape[1], y[i].shape[2])
                for j in range(imputation):
                    x[i] = np.append(x[i], a, axis=0)
                    y[i] = np.append(y[i], b, axis=0)

        #print(f'{filename}: {x.shape}')
        return x, y, sensors_num, int(time / timeWinNum)


if __name__ == '__main__':
    # 读取数据
    timeWinNum = 24  # 时间窗个数
    x, y, sensors_num, timewindows = readUcrTsv("data/CinCECGTorso/CinCECGTorso_TRAIN.tsv",
                                                            timeWinNum,
                                                            method=2)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    print(x[3].shape)
def readucr(filename): # Load data
    data = np.loadtxt(filename, dtype=str, delimiter=',')
    data = data[:, 0: -1]  # 去掉最后一个label
    truncate = np.int(np.floor((len(data[0]) - 1) / 8) * 8)
    x = np.array(data[:, 0: truncate], dtype=np.float32)
    y = np.array(data[:, 1: (truncate + 1)], dtype=np.float32)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1))
    print(f'{filename}: {x.shape}')
    return x, y


def readucr2(filename, sensors=1):
    # sensors 为传感器数量
    data = np.loadtxt(filename, dtype=str, delimiter=',')
    data = data[:, 0: -1]  # 去掉最后一个label
    truncate = np.int(np.floor((len(data[0]) - 1) / 8) * 8)
    x = np.array(data[:, 0:-1], dtype=np.float32)
    y = np.array(data[:, 1:], dtype=np.float32)
    timesteps = x.shape[1]
    x = x.reshape(-1, sensors, timesteps).swapaxes(2, 1)
    y = y.reshape(-1, sensors, timesteps).swapaxes(2, 1)
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
        max = np.max(x_train[i])
        min = np.min(x_train[i])
        k = 2 / (max - min)
        x_train[i] = -1 + k * (x_train[i] - min)
        # scaler = MinMaxScaler(feature_range=(-1, 1)) # 归一化到 [-1, 1] 区间内
        # x_train[i] = scaler.fit_transform(x_train[i]) # fit 获得最大值和最小值，transform 执行归一化
    return x_train


def is_abnormal(modelpath, X_test, X_train, Y_train):
    mini_batch_size = 6
    flag = 0 #标识是否有异常数据
    sigma = 1.0
    model = load_model(modelpath)

    abnormal_dict = {}
    index = 0
    X_test = np.repeat(X_test, 6, axis=0)

    [threshold, _] = model.evaluate(X_train, Y_train, batch_size=mini_batch_size, verbose=0)
    predict_value = model.predict(X_test)

    threshold = threshold * sigma
    print('threshold is: {}'.format(threshold))
    [loss, _] = model.evaluate(X_test[index * 6:(index + 1) * 6], X_test[index * 6:(index + 1) * 6],
            batch_size=mini_batch_size, verbose=0)
    print('loss is: {}'.format(loss))

    # 定位异常数据时间窗
    if loss > threshold:
        print('This is an abnormal data')
        while index < X_test.shape[2]:
            j = 0
            value = 0
            while j < X_test.shape[3]:
                value = value + (X_test[0][0][index][j] - predict_value[0][0][index][j]) ** 2
                j = j + 1
            value = value / X_test.shape[3]
            print('{} of loss is: {}'.format(index,loss))
            if value > threshold:
                abnormal_dict[index] = value
            index = index + 1
        print('the index of abnormal data is: {}'.format(abnormal_dict))
        flag = 1
    else:
        print('This is a normal data')

    return abnormal_dict if flag == 1 else None