from  cnn_AE_1 import Cnn_AE_1
import tensorflow as tf
import numpy as np
from cnn_AE_2 import Cnn_AE_2

def readucr(filename):#读取数据
    data = np.loadtxt(filename, dtype=str, delimiter = ',')
    x = data[:, 0:576]
    return x


def preprocess(x_train, x_test):
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    return train_dataset, test_dataset


if __name__ == '__main__':
    # [60, 576]
    x_train = readucr('data/Car_TRAIN.txt')
    x_train = np.expand_dims(x_train, -1)
    x_train = np.expand_dims(x_train, 1)
    x_train = x_train.astype(float)
    x_train = tf.cast(x_train, dtype=tf.float32)

    # [60, 576]
    x_test = readucr('data/Car_TEST.txt')
    x_test = np.expand_dims(x_test, -1)
    x_test = np.expand_dims(x_test, 1)
    x_test = x_test.astype(float)
    x_test = tf.cast(x_test, dtype=tf.float32)
    # train_dataset, test_dataset = preprocess(x_train, x_test)
    input_shape = (1, x_train.shape[2], 1)
    cnn_Auto = Cnn_AE_1('result', input_shape, True)
    # cnn_Auto = Cnn_AE_2('result', input_shape, True)
    cnn_Auto.fit_model(x_train, x_train, x_test, x_test)






    # x_train = data_Process.readucr('data/Car_TRAIN.txt', 1)  # 每一个是577个数据，一类有16个
    # x_train = np.array(x_train)
    # print(x_train.shape)
    # x_test = data_Process.readucr('data/Car_TEST.txt', 1)[0]
    # x_val = data_Process.readucr('data/Car_TEST.txt', 1)[1]
    # cnn_Auto = Cnn_AE_1('result', (len(x_train),), True)
    # cnn_Auto.train_model(x_train, x_train, x_val, x_val)