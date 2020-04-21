from data_utils import readucr, readucr2, Data_MinMax_Scaler,  readUcrTsv, readNab
from cnn_AE_2 import Cnn_AE_2
from cnn_AE_n import Cnn_AE_n
from cnn_AE_npron import Cnn_AE_npron
from cnn_AE_nproc import Cnn_AE_nproc
from GRU import Single_GRU, Multi_Gru, output_of_single_gru, output_of_multi_gru

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_AE_2 import Cnn_AE_2
# from tensorflow_core.examples.tutorials.mnist import input_data
import os

if __name__ == '__main__':
    # 读取数据
    timeWinNum = 24  # 时间窗个数
    # dataset_name = sys.argv[1]
    dataset_name = 'NAB/artificialNoAnomaly'
    dataset_path = os.path.join('data', dataset_name)
    x_train, y_train, sensors_num, timewindows = readNab(dataset_path, timeWinNum)
    x_test, y_test, _, _ = readNab(dataset_path, timeWinNum)

    # 将数据归一化
    # x_train = Data_MinMax_Scaler(x_train)
    # y_train = Data_MinMax_Scaler(y_train)
    # x_test = Data_MinMax_Scaler(x_test)
    # y_test = Data_MinMax_Scaler(y_test)
    # print(f'x_train after normalization:{x_train.shape}')
    # print(f'x_test after normalization :{x_train.shape}')

    gru_model = {}
    GruFilepath = {}
    gru_train = {}
    gru_test = {}

    # 训练gru模型
    for i in range(sensors_num):
        # '''
        gru_model['model_' + str(i)] = Multi_Gru(dataset_name,
                                                 x_train[i].shape[1:],
                                                 output_shape=1,
                                                 h_dim=20,
                                                 timewindows=timewindows,
                                                 verbose=True)
        gru_model['model_' + str(i)].fit(x_train[i], y_train[i], x_test[i], y_test[i], epochs=500)
        # '''
        # 提取gru的输出
        GruFilepath['model_' + str(i)] = gru_model['model_' + str(i)].model_path
        # GruFilepath['model_' + str(i)] = 'result/multigru/CinCECGTorso_gru_model_' + str(i) + '.hdf5'
        gru_train['model_' + str(i)] = output_of_multi_gru(x_train[i], GruFilepath['model_' + str(i)],
                                                           timewindows=timewindows)
        gru_test['model_' + str(i)] = output_of_multi_gru(x_test[i], GruFilepath['model_' + str(i)],
                                                          timewindows=timewindows)
        print(gru_train['model_' + str(i)].shape)
        print(gru_test['model_' + str(i)].shape)

    gru_train_all = [gru_train['model_' + str(i)] for i in range(sensors_num)]
    gru_test_all = [gru_test['model_' + str(i)] for i in range(sensors_num)]
    gru_train_concat = tf.concat(gru_train_all, axis=1)
    gru_test_concat = tf.concat(gru_test_all, axis=1)

    # 训练cnn模型
    input_shape = gru_train_concat.shape[1:]
    print(f"input_shape of CNN:{input_shape}")
    cnn_Auto = Cnn_AE_n('result', input_shape, 6, True)
    cnn_Auto.fit_model(gru_train_concat, gru_train_concat, gru_test_concat, gru_test_concat, epochs=500)
