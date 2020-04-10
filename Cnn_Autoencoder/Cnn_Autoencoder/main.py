#!/usr/bin/env python
#-*- coding: UTF-8 -*-
from data_utils import readucr, readucr2, Data_MinMax_Scaler, is_abnormal, readUcrTsv
from cnn_AE_1 import Cnn_AE_1
from cnn_AE_2 import Cnn_AE_2
from cnn_AE_n import Cnn_AE_n
from cnn_AE_npron import Cnn_AE_npron
from cnn_AE_nproc import Cnn_AE_nproc
from cnn_AE_1pro import Cnn_AE_1pro
# from GRU2 import Single_GRU, Multi_Gru,  output_of_single_gru, output_of_multi_gru
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_AE_2 import Cnn_AE_2
import os


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# 第一个想法，相同卷积核模型训练
def train_cnn_AE_n(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_n('result', input_shape, batchsize,verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)


# 第二个想法，不同卷积核模型训练，在传感器维度上拼接
def train_cnn_AE_npron(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_npron('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)


# 第三个想法，不同卷积核模型训练，在channel维度上拼接
def train_cnn_AE_nproc(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_nproc('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)


# 第四个想法，卷积核长宽不一致.（数据的rows不得少于4）
def train_cnn_AE_2(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_2('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)


if __name__ == '__main__':
    # （-1,1）之间的随机数
    mu = 0
    sigma = 1
    sample_slice = -1 + 2* np.random.random((4, 512, 20))
    # print(sample_slice[0, 0, :])
    # sample_slice_gauss = sample_slice + np.random.normal(mu, sigma, (4, 512, 20))/100
    # print(sample_slice_gauss[0, 0, :])
    # sample_slice_gauss_scaler = Data_MinMax_Scaler(np.expand_dims(sample_slice_gauss, 0))
    # print(sample_slice_gauss_scaler[0, 0, 0, :])
    np.save('sample.npy', sample_slice)
    sam = np.load('sample.npy')
    x_train_random = np.array([sample_slice + np.random.normal(mu, sigma, (4, 512, 20)) for i in range(1800)])
    x_val_random = np.array([sample_slice + np.random.normal(mu, sigma, (4, 512, 20)) for i in range(900)])
    # 归一化
    x_train_random = Data_MinMax_Scaler(x_train_random)
    x_val_random = Data_MinMax_Scaler(x_val_random)
    train_cnn_AE_n((4, 512, 20), x_train_random, x_train_random, x_val_random, x_val_random, epochs=1000)
    train_cnn_AE_npron((4, 512, 20), x_train_random, x_train_random, x_val_random, x_val_random, epochs=1000)
    cnn_AE_n_model = load_model('result/cnn_AE_n/best_model.hdf5', custom_objects={'root_mean_squared_error':root_mean_squared_error})
    cnn_AE_npron_model = load_model('result/cnn_AE_npron/best_model.hdf5',custom_objects={'root_mean_squared_error':root_mean_squared_error})
    # test_slice = np.tile(-1 + 2* np.random.random((4, 512, 20)), (6, 1, 1, 1))
    test_slice = np.tile(sample_slice, (6, 1, 1, 1))
    print(test_slice[0, 0, 0, :])
    recons_slice_n = cnn_AE_n_model.predict(test_slice)
    recons_slice_npron = cnn_AE_npron_model.predict(test_slice)
    print(recons_slice_n[0, 0, 0, :])
    print(recons_slice_npron[0, 0, 0, :])





