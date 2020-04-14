from data_utils import readucr, readucr2, Data_MinMax_Scaler, is_abnormal, readUcrTsv
from cnn_AE_2 import Cnn_AE_2
from cnn_AE_n import Cnn_AE_n
from cnn_AE_npron import Cnn_AE_npron
from cnn_AE_nproc import Cnn_AE_nproc
from GRU import Single_GRU, Multi_Gru,  output_of_single_gru, output_of_multi_gru

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_AE_2 import Cnn_AE_2
# from tensorflow_core.examples.tutorials.mnist import input_data
import os


if __name__ == '__main__':
    # 读取训练数据及测试数据
    timeWinNum = 32  # 时间窗个数
    # dataset_name = sys.argv[1]
    dataset_name = 'CinCECGTorso'
    dataset_path = os.path.join('data', dataset_name)
    x_train, y_train, sensors_num, timewindows = readUcrTsv(dataset_path + '/' + dataset_name + '_TEST.tsv', timeWinNum,
                                                            method=2)
    test_data, _, _, _ = readUcrTsv('data/CinCECGTorso/test_data.tsv', timeWinNum, method=2)

    # 提取gru的输出
    GruFilepath = {}
    gru_train = {}
    g_test = {}
    for i in range(sensors_num):

        GruFilepath['model_' + str(i)] = 'result/multigru/CinCECGTorso_gru_model_' + str(i) + '.hdf5'
        gru_train['model_' + str(i)] = output_of_multi_gru(x_train[i], GruFilepath['model_' + str(i)],
                                                           timewindows=timewindows)
        g_test['model_' + str(i)] = output_of_multi_gru(test_data[i], GruFilepath['model_' + str(i)],
                                                        timewindows=timewindows)
        print(gru_train['model_' + str(i)].shape)
        print(g_test['model_' + str(i)].shape)

    gru_train_all = [gru_train['model_' + str(i)] for i in range(sensors_num)]
    gru_train_concat = tf.concat(gru_train_all, axis=1)
    g_test_all = [g_test['model_' + str(i)] for i in range(sensors_num)]
    g_test_concat = tf.concat(g_test_all, axis=1)
    print(gru_train_concat.shape)
    print(g_test_concat.shape)

    # 测试是否异常
    cnnfilepath = 'result/cnn_AE_n/best_model.hdf5'
    is_abnormal(cnnfilepath, g_test_concat[:1], gru_train_concat, gru_train_concat)




