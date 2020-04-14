#!/usr/bin/env python
#-*- coding: UTF-8 -*-
import os
import time
import datetime
import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperParameters
from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, \
    Input, UpSampling2D, ZeroPadding1D, ZeroPadding2D, Lambda, \
    Conv2DTranspose, Activation, Cropping2D

# tf.compat.v1.disable_eager_execution()
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def abMaxPooling1D(inputs, pool_size=2, strides=2, padding='SAME'):
#     # tf.nn.max_pool(value, ksize, strides, padding, name=None)
#     # output1 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
#     # output2 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
#     output1 = tf.nn.max_pool1d(inputs, ksize=pool_size, strides=strides, padding=padding)
#     output2 = tf.nn.max_pool1d(-inputs, ksize=pool_size, strides=strides, padding=padding)
#     mask = output1 >= output2
#     output = tf.where(mask, output1, -output2)
#     return output
#
# def abMaxPooling2D(inputs, pool_size=[2, 2], strides=2, padding='SAME'):
#     # output1 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
#     # output2 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
#     output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
#     output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
#     mask = output1 >= output2
#     output = tf.where(mask, output1, -output2)
#     return output


def abMaxPooling_with_argmax(inputs, pool_size=2, strides=2, padding='SAME'):

    output1, argmax1 = tf.nn.max_pool_with_argmax(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2, argmax2 = tf.nn.max_pool_with_argmax(-inputs, ksize=pool_size, strides=strides, padding=padding)
    argmax1 = tf.stop_gradient(argmax1)
    argmax2 = tf.stop_gradient(argmax2)
    # output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
    # output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    argmax = tf.where(mask, argmax1, argmax2)
    return (output, argmax)

def unAbMaxPooling(inputs_argmax, ksize, strides=2, padding='SAME'):
    # 假定 ksize = strides
    inputs = inputs_argmax[0]
    argmax = inputs_argmax[1]
    input_shape = inputs.get_shape()
    if padding == 'SAME':
        rows = input_shape[1] * ksize[1]
        cols = input_shape[2] * ksize[2]
    else:
        rows = (input_shape[1] - 1) * ksize[1] + ksize[1]
        cols = (input_shape[2] - 1) * ksize[2] + ksize[2]
    # 计算new shape
    output_shape = (input_shape[0], rows, cols, input_shape[3])
    # 计算索引
    one_like_mask = tf.ones_like(argmax)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = argmax // (output_shape[2] * output_shape[3])
    x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    c = one_like_mask * feature_range
    # 转置索引
    update_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, update_size]))
    values = tf.reshape(inputs, [update_size])
    outputs = tf.scatter_nd(indices, values, output_shape)
    return outputs


# def reshapes(x, retype):
#     if retype == 'reshapedim':
#         x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
#     if retype == 'squeeze':
#         x = tf.squeeze(x, [1])
#     return x

def reshapes(x):
    x = tf.squeeze(x, [1])
    x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
    return x

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class Cnn_AE_n(HyperModel):
    def __init__(self, output_directory, input_shape, batchsize, verbose=False, **kwargs):
        super(Cnn_AE_n, self).__init__(**kwargs)
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.batchsize = batchsize
        self.set_Config()
        # hp = HyperParameters()
        # self.model = self.build_model()
        # verbose是信息展示模式
        # if verbose == True:
        #     self.model.summary()
        self.verbose = verbose


    def set_Config(self):
        self.conv1_filters = 64
        self.conv1_kersize = 3
        self.conv2_filters = 32
        self.conv2_kersize = 3
        self.conv3_filters = 16
        self.conv3_kersize = 3
        self.conv4_filters = 8
        self.conv4_kersize = 3
        self.z_filters = self.conv4_filters
        self.z_kersize = 3


    def set_ModCallbacks(self):
        file_dir = os.path.join(self.output_directory, 'cnn_AE_n')
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file_path = os.path.join(file_dir, 'best_model.hdf5')

        log_dir = ".\log\\fit\\cnn_AE_n\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='val_loss',
                                                           save_best_only=True,
                                                           mode='auto')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
                                                      min_lr=0.0001)

        self.callbacks = [tensorboard, model_checkpoint, reduce_lr]
        return self.callbacks


    def set_HpCallbacks(self):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
                                                      min_lr=0.0001)
        callbacks = [reduce_lr]
        return callbacks


    def build(self, hp):
        # input --> (None, n, 576, c)
        input_layer = Input(batch_shape=(self.batchsize, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        # Encoder
        # conv block -1 （卷积+池化）
        self.conv1_filters = hp.Int('conv1_filters', 32, 128, step=16)
        self.conv1_kersize = hp.Int('conv1_kersize', 3, 7, step=2)
        conv1 = ZeroPadding2D((0, self.conv1_kersize//2))(input_layer)
        h1 = input_layer.get_shape()[1]
        conv1 = Conv2D(filters=self.conv1_filters, kernel_size=(h1, self.conv1_kersize))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)
        conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool1')(conv1)
        conv1_pool = Lambda(reshapes, name='reshape1')(conv1_pool)


        # conv block -2 （卷积+池化）
        self.conv2_filters = hp.Int('conv2_filters', 32, 64, step=2)
        self.conv2_kersize = hp.Int('conv2_kersize', 3, 7, step=2)
        conv2 = ZeroPadding2D((0, self.conv2_kersize//2))(conv1_pool)
        h2 = conv2.get_shape()[1]
        conv2 = Conv2D(filters=self.conv2_filters, kernel_size=(h2, self.conv2_kersize))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation='relu')(conv2)
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool2')(conv2)
        conv2_pool = Lambda(reshapes, name='reshape2')(conv2_pool)

        # conv block -3 （卷积+池化）
        self.conv3_filters = hp.Int('conv3_filters', 16, 64, step=2)
        self.conv3_kersize = hp.Int('conv3_kersize', 3, 7, step=2)
        conv3 = ZeroPadding2D((0, self.conv3_kersize//2))(conv2_pool)
        h3 = conv3.get_shape()[1]
        conv3 = Conv2D(filters=self.conv3_filters, kernel_size=(h3, self.conv3_kersize))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation='relu')(conv3)
        conv3_pool, conv3_argmax  = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool3')(conv3)
        conv3_pool = Lambda(reshapes, name='reshape3')(conv3_pool)

        # conv block -4 （卷积+池化）
        self.conv4_filters = hp.Int('conv4_filters', 8, 32, step=2)
        self.conv4_kersize = hp.Int('conv4_kersize', 3, 7, step=2)
        conv4 = ZeroPadding2D((0, self.conv4_kersize//2))(conv3_pool)
        h4 = conv4.get_shape()[1]
        conv4 = Conv2D(filters=self.conv4_filters, kernel_size=(h4, self.conv4_kersize))(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation='relu')(conv4)
        conv4_pool, conv4_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool4')(conv4)
        conv4_pool = Lambda(reshapes, name='reshape4')(conv4_pool)

        # 中间层
        self.z_kersize = hp.Int('z_kersize', 3, 7, step=2)
        h_z = conv4_pool.get_shape()[1]
        z = ZeroPadding2D((0, self.z_kersize//2))(conv4_pool)
        z = Conv2D(filters=self.conv4_filters, kernel_size=(h_z, self.z_kersize))(z)
        z = BatchNormalization()(z)
        encoder = Activation(activation='relu')(z)


        # decoder
        # conv block -1 （反池化+反卷积）
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool1')([encoder, conv4_argmax])
        deconv1 = Conv2DTranspose(filters=self.conv3_filters, kernel_size=(h4, self.conv4_kersize), padding='same')(deconv1_unpool)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(activation='relu')(deconv1)

        # conv block -2 （反池化+反卷积）
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool2')([deconv1, conv3_argmax])
        deconv2 = Conv2DTranspose(filters=self.conv2_filters, kernel_size=(h3, self.conv3_kersize), padding='same')(deconv2_unpool)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(activation='relu')(deconv2)

        # conv block -3 （反池化+反卷积）
        deconv3_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool3')([deconv2, conv2_argmax])
        deconv3 = Conv2DTranspose(filters=self.conv1_filters, kernel_size=(h2, self.conv2_kersize), padding='same')(deconv3_unpool)
        # deconv3 = Cropping2D(cropping=((0, 0), (1, 1)))(deconv3)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Activation(activation='relu')(deconv3)

        # conv block -4 （反池化+反卷积）
        deconv4_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool4')([deconv3, conv1_argmax])
        deconv4 = Conv2DTranspose(filters=self.input_shape[2], kernel_size=(h1, self.conv1_kersize), padding='valid')(deconv4_unpool)
        deconv4 = Cropping2D(cropping=((0, 0), (self.conv1_kersize//2, self.conv1_kersize//2)))(deconv4)
        deconv4 = BatchNormalization()(deconv4)
        output_layer = Activation(activation='tanh')(deconv4)



        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=root_mean_squared_error,
                      optimizer=optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      metrics=[root_mean_squared_error],
                      experimental_run_tf_function=False)

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, epochs):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        # batch_size = 12
        self.set_ModCallbacks()
        nb_epochs = epochs

        # 小批量训练大小
        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        batch_size = 6

        # 开始时间
        start_time = time.time()

        # file_path = os.path.join(self.output_directory, 'cnn_AE_n/best_model.hdf5')

        # 训练模型
        hist = self.model.fit(x_train, y_train,
                              epochs=nb_epochs,
                              verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks)

        # print('history：', hist.history)
        # 训练持续时间
        duration = time.time() - start_time
        print('duration: ', duration)
        keras.backend.clear_session()

        # 做测试，所以需要加载模型
        # model = load_model(file_path)
        # loss = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        # print('test_loss: ', loss)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

