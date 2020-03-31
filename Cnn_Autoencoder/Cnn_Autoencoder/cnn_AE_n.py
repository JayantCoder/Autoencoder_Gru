# Import all the required Libraries
import os
import time
import datetime
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
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

# def reshape_output_shape(input_shape):
#     if len(input_shape) == 4:
#         return (input_shape[0], input_shape[2], input_shape[3])
#     if len(input_shape) == 3:
#         return (input_shape[0], input_shape[2], input_shape[1], 1)


class Cnn_AE_n:
    def __init__(self, output_directory, input_shape, batchsize, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, batchsize)
        # verbose是信息展示模式
        if verbose == True:
            self.model.summary()
        self.verbose = verbose

    def build_model(self, input_shape, batchsize):
        # input --> (None, n, 576, c)
        input_layer = Input(batch_shape=(batchsize, input_shape[0], input_shape[1], input_shape[2]))
        # Encoder
        # conv block -1 （卷积+池化）
        conv1 = ZeroPadding2D((0, 1))(input_layer)
        h1 = input_layer.get_shape()[1]
        conv1 = Conv2D(filters=16, kernel_size=(h1, 3))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)
        conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool1')(conv1)
        conv1_pool = Lambda(reshapes, name='reshape1')(conv1_pool)


        # conv block -2 （卷积+池化）
        conv2 = ZeroPadding2D((0, 1))(conv1_pool)
        h2 = conv2.get_shape()[1]
        conv2 = Conv2D(filters=8, kernel_size=(h2, 3))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation='relu')(conv2)
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool2')(conv2)
        conv2_pool = Lambda(reshapes, name='reshape2')(conv2_pool)

        # conv block -3 （卷积+池化）
        conv3 = ZeroPadding2D((0, 1))(conv2_pool)
        h3 = conv3.get_shape()[1]
        conv3 = Conv2D(filters=8, kernel_size=(h3, 3))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation='relu')(conv3)
        conv3_pool, conv3_argmax  = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool3')(conv3)
        conv3_pool = Lambda(reshapes, name='reshape3')(conv3_pool)

        # 中间层
        z = ZeroPadding2D((0, 1))(conv3_pool)
        z = Conv2D(filters=8, kernel_size=(conv3_pool.get_shape()[1], 3))(z)
        z = BatchNormalization()(z)
        encoder = Activation(activation='relu')(z)


        # decoder
        # conv block -1 （反卷积+反池化）
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool1')([encoder, conv3_argmax])
        deconv1 = Conv2DTranspose(filters=8, kernel_size=(h3, 3), padding='same')(deconv1_unpool)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(activation='relu')(deconv1)

        # conv block -2 （反卷积+反池化）
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool2')([deconv1, conv2_argmax])
        deconv2 = Conv2DTranspose(filters=16, kernel_size=(h2, 3), padding='same')(deconv2_unpool)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(activation='relu')(deconv2)

        # conv block -3 （反卷积+反池化）
        deconv3_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool3')([deconv2, conv1_argmax])
        deconv3 = Conv2DTranspose(filters=input_shape[2], kernel_size=(h1, 3), padding='valid')(deconv3_unpool)
        deconv3 = Cropping2D(cropping=((0, 0), (1, 1)))(deconv3)
        deconv3 = BatchNormalization()(deconv3)
        output_layer = Activation(activation='tanh')(deconv3)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mse',
                      optimizer=optimizers.Adam(0.001),
                      metrics=['mse'],
                      experimental_run_tf_function=False)

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
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=20,
                                                      min_lr=0.0001)

        self.callbacks = [tensorboard, model_checkpoint, reduce_lr]
        # self.callbacks = [model_checkpoint, reduce_lr]

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, epochs):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        # batch_size = 12
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
