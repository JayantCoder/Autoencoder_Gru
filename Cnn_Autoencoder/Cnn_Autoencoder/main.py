from  cnn_AE_1 import Cnn_AE_1
from cnn_AE_1pro import Cnn_AE_1pro
import tensorflow as tf
import numpy as np
from cnn_AE_2 import Cnn_AE_2
from tensorflow_core.examples.tutorials.mnist import input_data

def readucr(filename):#读取数据
    data = np.loadtxt(filename, dtype=str, delimiter = ',')
    x = data[:, 0:576]
    return x

    # mnist = input_data.read_data_sets(filename, one_hot=True)
    # trainimg = mnist.train.images
    # testimg = mnist.test.images
    # return trainimg, testimg


# def preprocess(x_train, x_test):
#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
#     test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
#     return train_dataset, test_dataset


if __name__ == '__main__':
    # [60, 576]
    x_train = readucr('data/Car/Car_TRAIN.txt')
    x_train = np.expand_dims(x_train, -1)
    x_train = np.expand_dims(x_train, 1)
    x_train = x_train.astype(float)
    x_train = tf.cast(x_train, dtype=tf.float32)

    # [60, 576]
    x_test = readucr('data/Car/Car_TEST.txt')
    x_test = np.expand_dims(x_test, -1)
    x_test = np.expand_dims(x_test, 1)
    x_test = x_test.astype(float)
    x_test = tf.cast(x_test, dtype=tf.float32)
    # train_dataset, test_dataset = preprocess(x_train, x_test)
    input_shape = (1, x_train.get_shape()[2], 1)

    # 图片数据测试
    # x_train, x_test = readucr(r'data/mnist/')
    # x_train = np.reshape(x_train, (-1, 28, 28))
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.reshape(x_test, (-1, 28, 28))
    # x_test = np.expand_dims(x_test, -1)
    # input_shape = (28, 28, 1)

    # cnn_Auto = Cnn_AE_1('result', input_shape, True)
    cnn_Auto = Cnn_AE_1pro('result', input_shape, True)
    # cnn_Auto = Cnn_AE_2('result', input_shape, True)
    cnn_Auto.fit_model(x_train, x_train, x_test, x_test)
    # cnn_Auto.fit_model(train_dataset, test_dataset)



