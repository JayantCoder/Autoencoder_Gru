# Import all the required Libraries
import os
import time, datetime
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, \
    Input, UpSampling2D, Lambda, Conv2DTranspose, Activation, ZeroPadding2D,\
    Cropping2D
# from keras.layers.advanced_activations import LeakyReLU
import keras.optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def abMaxPooling1D(inputs, pool_size=2, strides=2, padding='SAME'):
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
    # inputs = inputs_argmax[0]
    # argmax = inputs_argmax[1]
    # input_shape = inputs.get_shape()
    # if padding == 'SAME':
    #     rows = input_shape[1] * ksize[1]
    #     cols = input_shape[2] * ksize[2]
    # else:
    #     rows = (input_shape[1] - 1) * ksize[1] + ksize[1]
    #     cols = (input_shape[2] - 1) * ksize[2] + ksize[2]
    # 假定 ksize != strides
    inputs = inputs_argmax[0]
    argmax = inputs_argmax[1]
    input_shape = inputs.get_shape()
    if padding == 'SAME':
        # strides = [1, 2], ksize = [2, 2]的情况，且数据在这种情况下没有padding
        rows = input_shape[1] - 1 + ksize[1]
        cols = input_shape[2] * ksize[2]
    else:
        # 这个是错的
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
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class Cnn_AE_2:
    def __init__(self, output_directory, input_shape, batchsize, verbose=False):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.batchsize = batchsize
        self.set_Config()
        self.model = self.build_model()
        if verbose == True:
            self.model.summary()
        self.verbose = verbose


    def set_Config(self):
        self.conv1_filters = 64
        self.conv1_kersize = (3, 5)
        self.conv2_filters = 32
        self.conv2_kersize = (3, 5)
        self.conv3_filters = 16
        self.conv3_kersize = (3, 5)
        self.conv4_filters = 8
        self.conv4_kersize = (3, 5)
        self.z_filters = self.conv4_filters
        self.z_kersize = (3, 5)


    def set_ModCallbacks(self):
        file_dir = os.path.join(self.output_directory, 'cnn_AE_2')
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file_path = os.path.join(file_dir, 'best_model.hdf5')

        log_dir = ".\log\\fit\\cnn_AE_2\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='val_loss',
                                                           save_best_only=True,
                                                           mode='auto')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                                      min_lr=0.0001)

        self.callbacks = [tensorboard, model_checkpoint, reduce_lr]
        return self.callbacks


    def build_model(self):
        input_layer = Input(batch_shape=(self.batchsize, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

        # conv block -1 （卷积+池化）
        conv1 = Conv2D(filters=self.conv1_filters, kernel_size=self.conv1_kersize, padding='same')(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)
        conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax,
                                          arguments={'pool_size': [2, 2], 'strides': [1, 2], 'padding': 'VALID'},
                                          name='abMaxPool1')(conv1)


        # conv block -2 （卷积+池化）
        conv2 = Conv2D(filters=self.conv2_filters, kernel_size=self.conv2_kersize, padding='same')(conv1_pool)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation='relu')(conv2)
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax,
                                          arguments={'pool_size': [2, 2], 'strides': [1, 2],
                                                     'padding': 'VALID'}, name='abMaxPool2')(conv2)
        # conv2_pool = Lambda(abMaxPooling2D, arguments={'pool_size': [2, 2]}, name='abMaxPool2')(conv2)

        # conv block -3 （卷积+池化）
        conv3 = Conv2D(filters=self.conv3_filters, kernel_size=self.conv3_kersize, padding='same')(conv2_pool)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation='relu')(conv3)
        conv3_pool, conv3_argmax = Lambda(abMaxPooling_with_argmax,
                                          arguments={'pool_size': [2, 2], 'strides': [1, 2], 'padding': 'VALID'},
                                          name='abMaxPool3')(conv3)

        # conv block -4 （卷积+池化）
        conv4 = Conv2D(filters=self.conv4_filters, kernel_size=self.conv4_kersize, padding='same')(conv3_pool)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation='relu')(conv4)
        conv4_pool, conv4_argmax = Lambda(abMaxPooling_with_argmax,
                                          arguments={'pool_size': [1, 2], 'strides': [1, 2], 'padding': 'VALID'},
                                          name='abMaxPool4')(conv4)

        # 中间层
        z = Conv2D(filters=self.z_filters, kernel_size=self.z_kersize, padding='same')(conv4_pool)
        z = BatchNormalization()(z)
        encoder = Activation(activation='relu')(z)

        # decoder
        # conv block -1 （反卷积+反池化）
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]},
                                name='unAbPool1')([encoder, conv4_argmax])
        deconv1 = Conv2DTranspose(filters=self.conv3_filters, kernel_size=self.conv4_kersize, padding='same')(deconv1_unpool)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(activation='relu')(deconv1)


        # conv block -2 （反卷积+反池化）
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 2, 2, 1]},
                                name='unAbPool2')([deconv1, conv3_argmax])
        deconv2 = Conv2DTranspose(filters=self.conv2_filters, kernel_size=self.conv3_kersize, padding='same')(deconv2_unpool)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(activation='relu')(deconv2)

        # conv block -3 （反卷积+反池化）
        deconv3_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 2, 2, 1]},
                                name='unAbPool3')([deconv2, conv2_argmax])
        deconv3 = Conv2DTranspose(filters=self.conv1_filters, kernel_size=self.conv2_kersize, padding='same')(deconv3_unpool)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Activation(activation='relu')(deconv3)

        # conv block -4 （反卷积+反池化）
        deconv4_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 2, 2, 1]},
                                name='unAbPool4')([deconv3, conv1_argmax])
        deconv4 = Conv2DTranspose(filters=self.input_shape[2], kernel_size=self.conv1_kersize, padding='same')(deconv4_unpool)
        deconv4 = BatchNormalization()(deconv4)
        output_layer = Activation(activation='tanh')(deconv4)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=root_mean_squared_error,
                      optimizer=optimizers.Adam(0.001),
                      metrics=[root_mean_squared_error])
        return model

    def fit_model(self, x_train, y_train, x_val, y_val, epochs):

        # batch_size = 6
        self.set_ModCallbacks()
        nb_epochs = epochs

        # 小批量训练大小
        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        # 训练模型
        hist = self.model.fit(x_train, y_train,
                              epochs=nb_epochs,
                              verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks)

        duration = time.time() - start_time
        print(duration)
        keras.backend.clear_session()

        # 做测试，所以需要加载模型
        # model = load_model(os.path.join(self.output_directory, 'cnn_AE_2/best_model.hdf5'))
        #
        # loss = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        # y_pred = model.predict(x_val)
        # y_pred = np.argmax(y_pred, axis=1)
        # print('test_loss: ', loss)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)