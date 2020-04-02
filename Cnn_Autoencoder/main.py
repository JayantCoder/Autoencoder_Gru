from data_utils import readucr, readucr2, Data_MinMax_Scaler, is_abnormal, readUcrTsv
from cnn_AE_1 import Cnn_AE_1
from cnn_AE_2 import Cnn_AE_2
from cnn_AE_n import Cnn_AE_n
from cnn_AE_npron import Cnn_AE_npron
from cnn_AE_nproc import Cnn_AE_nproc
from cnn_AE_1pro import Cnn_AE_1pro
from GRU2 import Single_GRU, Multi_Gru,  output_of_single_gru, output_of_multi_gru
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_AE_2 import Cnn_AE_2
# from tensorflow_core.examples.tutorials.mnist import input_data
import os

# 第一个想法，相同卷积核模型训练
def train_cnn_AE_n(input_shape, x_train, y_train, x_test, y_test, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_n('result', input_shape, batchsize,verbose)
    model.fit_model(x_train, y_train, x_test, y_test, epochs)


# 第二个想法，不同卷积核模型训练，在传感器维度上拼接
def train_cnn_AE_npron(input_shape, x_train, y_train, x_test, y_test, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_npron('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_test, y_test, epochs)


# 第三个想法，不同卷积核模型训练，在channel维度上拼接
def train_cnn_AE_nproc(input_shape, x_train, y_train, x_test, y_test, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_nproc('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_test, y_test, epochs)


# 第四个想法，卷积核长宽不一致.（数据的rows不得少于4）
def train_cnn_AE_2(input_shape, x_train, y_train, x_test, y_test, batchsize=6, epochs=400, verbose=True):
    model = Cnn_AE_2('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_test, y_test, epochs)




if __name__ == '__main__':
    # 读取数据
    timeWinNum = 24  # 时间窗个数
    # dataset_name = sys.argv[1]
    dataset_name = 'CinCECGTorso'
    dataset_path = os.path.join('data', dataset_name)
    x_train, y_train, sensors_num, timewindows = readUcrTsv(dataset_path + '/' + dataset_name + '_TRAIN.tsv', timeWinNum,
                                                            method=2)
    x_test, y_test, _, _ = readUcrTsv(dataset_path + '/' + dataset_name + '_TEST.tsv', timeWinNum, method=2)
    print(x_train[0].shape)
    # 将数据归一化
    # x_train = Data_MinMax_Scaler(x_train)
    # y_train = Data_MinMax_Scaler(y_train)
    # x_test = Data_MinMax_Scaler(x_test)
    # y_test = Data_MinMax_Scaler(y_test)
    # print(f'x_train after normalization:{x_train.shape}')
    # print(f'x_test after normalization :{x_train.shape}')

    # 训练gru模型
    gru_model = {}
    GruFilepath = {}
    gru_train = {}
    gru_test = {}
    for i in range(sensors_num):
        print('-----------------------------------')
        print(f'SENSOR_{i + 1}')

        gru_model['model_' + str(i)] = Multi_Gru(dataset_name,
                                                 x_train[i].shape[1:],
                                                 output_shape=1,
                                                 h_dim=20,
                                                 timewindows=timewindows,
                                                 verbose=True)
        gru_model['model_' + str(i)].fit(x_train[i], y_train[i], x_test[i], y_test[i], epochs=500)

        # 提取gru的输出
        GruFilepath['model_' + str(i)] = gru_model['model_' + str(i)].model_path
        gru_train['model_' + str(i)] = output_of_multi_gru(x_train[i], GruFilepath['model_' + str(i)],
                                                           timewindows=timewindows)
        gru_test['model_' + str(i)] = output_of_multi_gru(x_test[i], GruFilepath['model_' + str(i)],
                                                          timewindows=timewindows)
        print("经Gru压缩后的output如下")
        print(f"gru_train: {gru_train['model_' + str(i)].shape}")
        print(f"gru_val: {gru_test['model_' + str(i)].shape}")

    gru_train_all = [gru_train['model_' + str(i)] for i in range(sensors_num)]
    gru_test_all = [gru_test['model_' + str(i)] for i in range(sensors_num)]
    gru_train_concat = tf.concat(gru_train_all, axis=1)
    gru_test_concat = tf.concat(gru_test_all, axis=1)
    print('===================================')
    print('GRU OK!')
    print('===================================')
    # 训练cnn模型
    input_shape = gru_train_concat.shape[1:]
    print("\n拼接后的shape: ")
    print(gru_train_concat.shape)
    train_cnn_AE_2(input_shape,
                       gru_train_concat, gru_train_concat,
                       gru_test_concat, gru_test_concat,
                       batchsize=6,
                       epochs=500,
                       verbose=True)
    print('===================================')
    print('第4种 CNN OK!')
    print('===================================')

    # 输入测试数据
    # test_data, _, _, _ = readUcrTsv(dataset_path + '/' + dataset_name + '_TEST.tsv', timeWinNum)
    # test_data = test_data[0:3, :]
    # g_test = output_of_multi_gru(test_data, GruFilepath, timewindows=timewindows)
    # print(g_test.shape)

    # 测试是否异常
    # cnnfilepath = cnn_Auto.model_path
    # is_abnormal(cnnfilepath, g_test, gru_train, gru_train)




