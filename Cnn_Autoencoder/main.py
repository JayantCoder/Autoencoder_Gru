from data_utils import readucr, Data_MinMax_Scaler,is_abnormal
from  cnn_AE_1 import Cnn_AE_1
from cnn_AE_1pro import Cnn_AE_1pro
from GRU import Autoencoder_GRU, output_of_gru
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_AE_2 import Cnn_AE_2
# from tensorflow_core.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    # 读取数据
    x_train, y_train = readucr('data/FordA/FordA_TRAIN.txt')
    x_test, y_test = readucr('data/FordA/FordA_TEST.txt')

    # 将数据归一化
    x_train = Data_MinMax_Scaler(x_train)
    y_train = Data_MinMax_Scaler(y_train)
    x_test = Data_MinMax_Scaler(x_test)
    y_test = Data_MinMax_Scaler(y_test)

    # 划分 训练集 和 验证集
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
    #                                                       test_size=0.33,
    #                                                       shuffle=True,
    #                                                       random_state=42)

    # 训练gru模型
    #gru_model = Autoencoder_GRU('result', x_train.shape[1:], 1)
    #gru_model.fit(x_train, y_train, x_test, y_test, epochs=500)

    # 提取gru的输出
    GruFilepath = 'result/gru/FordA_gru_model.hdf5'
    gru_train = output_of_gru(x_train, GruFilepath)
    gru_test = output_of_gru(x_test, GruFilepath)

    # 训练cnn模型
    #input_shape = (1, gru_train.shape[2], gru_train.shape[3])
    #cnn_Auto = Cnn_AE_1pro('result', input_shape, True)
    #cnn_Auto.fit_model(gru_train, gru_train, gru_test, gru_test, epochs=400)

    # 输入测试数据
    test_data, _ = readucr('data/FordA/test_data.txt')
    test_data = np.array(test_data[:6, :])
    g_test = output_of_gru(test_data, GruFilepath)

    # 测试是否异常
    cnnfilepath = 'result/cnn_AE_1pro/FordA_cnn_model.hdf5'
    print(gru_train.shape)
    gru_train = gru_train[:3600,:]
    is_abnormal(cnnfilepath, g_test, gru_train, gru_train)



