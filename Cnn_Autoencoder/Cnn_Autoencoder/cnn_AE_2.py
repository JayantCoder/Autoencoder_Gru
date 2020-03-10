# Import all the required Libraries
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, \
    Input, UpSampling2D, Lambda, Conv2DTranspose, Activation
# from keras.layers.advanced_activations import LeakyReLU
import keras.optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def abMaxPooling1D(inputs, pool_size=2, strides=2, padding='SAME'):
    # output1 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    # output2 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
    output1 = tf.nn.max_pool1d(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2 = tf.nn.max_pool1d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    return output

def abMaxPooling2D(inputs, pool_size=[2, 2], strides=2, padding='SAME'):
    # output1 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    # output2 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
    output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    return output

def abMaxPooling_with_argmax(inputs, pool_size=2, strides=2, padding='SAME'):
    output1, argmax1 = tf.nn.max_pool_with_argmax(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2, argmax2 = tf.nn.max_pool_with_argmax(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    argmax = tf.where(mask, argmax1, argmax2)
    return (output, argmax)


def unAbMaxPooling(inputs, argmax, strides):
    """
        input_tensor: unAbMaxPooling后的output tensor
        argmax：之前AbMaxPooling后最大值的index tensor
        strides：池化步长
        """
    #  计算input_shape
    input_shape = [x for x in inputs.get_shape()]  # [1, ,2, 2, 2]
    #  计算output_shape
    output_shape = [input_shape[0], input_shape[1] * strides, input_shape[2] * strides,
                    input_shape[3]]  # [1, 4, 4, 2]

    """
    The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
    flattened index ((b * height + y) * width + x) * channels + c
    """
    # 计算 b, y, x, c
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64),
                             shape=[input_shape[0], 1, 1, 1])  # batch index range
    # [1, 1, 1, 1]
    channel_range = tf.range(output_shape[3], dtype=tf.int64)

    b = tf.ones_like(argmax) * batch_range  # [1, 2, 2, 2]  广播

    y = argmax // (output_shape[2] * output_shape[3])  # [1, 2, 2, 2]  地板除以 width * channels ，不用管x， x一定比width小

    x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]  # [1, 2, 2, 2]

    c = tf.ones_like(argmax) * channel_range  # [1, 2, 2, 2] 广播

    # 计算input_tensor的元素个数
    updates_size = tf.size(inputs)  # 8

    # 将b, y, x, c堆叠起来 构成一个新 tensor ，维度 + 1
    # 索引矩阵 跟 b 对齐再转置
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, updates_size]))  # [8, 4]  4为 size of [b, y, x, c]
    # max数组
    values = tf.reshape(inputs, [updates_size, ])  # [8]

    # 重构 output 先构建全0的Tensor 然后value根据index中提供的坐标找到自己的位置
    outputs = tf.tensor_scatter_nd_add(tf.zeros(output_shape, values.dtype), indices, values)

    return outputs


def reshapes(x, retype):
    if retype == 'reshapedim':
        x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
    if retype == 'squeeze':
        x = tf.squeeze(x, [1])
    return x



class Cnn_AE_2:
    def __init__(self, output_directory, input_shape, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape)
        if verbose == True:
            self.model.summary()
        self.verbose = verbose


    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        input_layer = Input(shape=(32, 576, 1))
        # Encoder
        # conv block -1 （卷积+池化）
        if len(input_layer.shape) == 3:
            # conv1 = ZeroPadding1D(2)(input_layer)
            conv1 = Conv1D(filters=16, kernel_size=5, padding='same')(input_layer)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation(activation='relu')(conv1)
            conv1_pool = Lambda(abMaxPooling1D, arguments={'pool_size': 2})(conv1)
            conv1_pool = Lambda(reshapes, arguments={'retype': 'reshapedim'})(conv1_pool)
        if len(input_layer.shape) == 4:
            conv1 = Conv2D(filters=16, kernel_size=(3, 5), padding='same')(input_layer)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation(activation='relu')(conv1)
            conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [2, 2]}, name='abMaxPool1')(conv1)
            # conv1_pool = Lambda(abMaxPooling2D, arguments={'pool_size': [2, 2]})(conv1)

        # conv block -2 （卷积+池化）
        conv2 = Conv2D(filters=8, kernel_size=(3, 5), padding='same')(conv1_pool)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation='relu')(conv2)
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [2, 2]}, name='abMaxPool2')(conv2)

        # conv block -3 （卷积）
        conv3 = Conv2D(filters=8, kernel_size=(3, 5), padding='same')(conv2_pool)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation='relu')(conv3)
        encoder, conv3_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [2, 2]}, name='abMaxPool3')(conv3)

        # decoder
        # conv block -1 （反卷积+反池化）
        # deconv1_unpool = UpSampling2D(size=(2, 2))(encoder)
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'argmax': conv3_argmax, 'strides': 2}, name='unAbPool1')(encoder)
        deconv1 = Conv2DTranspose(filters=8, kernel_size=(3, 5), padding='same')(deconv1_unpool)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(activation='relu')(deconv1)

        # conv block -2 （反卷积+反池化）
        # deconv2_unpool = UpSampling2D(size=(2, 2))(deconv1)
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'argmax': conv2_argmax, 'strides': 2}, name='unAbPool2')(deconv1)
        deconv2 = Conv2DTranspose(filters=8, kernel_size=(3, 5), padding='same')(deconv2_unpool)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(activation='relu')(deconv2)

        # conv block -3 （反卷积+反池化）
        # deconv3 = Conv2D(filters=16, kernel_size=(8, 3), activation='relu', padding='same')(deconv2_unpool)
        # deconv3_unpool = UpSampling2D(size=(2, 2))(deconv2)
        deconv3_unpool = Lambda(unAbMaxPooling, arguments={'argmax': conv1_argmax, 'strides': 2}, name='unAbPool3')(deconv2)
        deconv3 = Conv2DTranspose(filters=16, kernel_size=(3, 5), padding='same')(deconv3_unpool)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Activation(activation='relu')(deconv3)

        # decoder = Conv2D(filters=1, kernel_size=(16, 5), activation='relu', padding='same')(deconv3_unpool)
        output_layer = Conv2DTranspose(filters=1, kernel_size=(3, 5), padding='same')(deconv3)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation(activation='sigmoid')(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(loss='mse', optimizer=optimizers.Adam(0.001), metrics=['mse'])

        file_path = os.path.join(self.output_directory, 'best_model.hdf5')

        # tensorboad = keras.callbacks.Tensorboard(log_dir='log')
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='val_loss', save_best_only=True, mode='auto')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=20,
                                                      min_lr=0.0001)

        # self.callbacks = [tensorboard, model_checkpoint]
        self.callbacks = [model_checkpoint, reduce_lr]

        return model

    def fit_model(self, x_train, y_train, x_val, y_val):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 12
        nb_epochs = 100

        # 小批量训练大小
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        # 训练模型
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                       verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks, validation_freq=4)

        duration = time.time() - start_time
        print(duration)

        # 做测试，所以需要加载模型
        model = load_model(os.path.join(self.output_directory, 'best_model.hdf5'))

        loss = model.evaluate(x_val, y_val, batch_size=mini_batch_size, verbose=0)
        # y_pred = model.predict(x_val)
        # y_pred = np.argmax(y_pred, axis=1)
        print('test_loss: ', loss)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()

    # def build_model(self, input_shape):
    #     # input
    #     input_layer = keras.layers.Input(input_shape)
    #
    #     # encoder
    #     # conv block -1 （卷积+池化）
    #     if len(K.int_shape(input_layer)) <= 1:
    #         print("输入数据有误！")
    #         return
    #     elif len(K.int_shape(input_layer)) == 2:
    #         conv1 = Conv1D(filters=512, kernel_size=5, strides=1, padding='same')(input_layer)
    #     else:
    #         conv1 = Conv2D(filters=512, kernel_size=(2, 5), strides=1, padding='same')(
    #             input_layer)
    #     # conv1 = Conv2D(filters=512, kernel_size=(2, 5), strides=1, padding='same')(input_layer)
    #     conv1 = keras.layers.PReLU(shared_axes=[1, 2])(conv1)
    #     conv1_pool = AbMaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    #     # conv block -2 （卷积+池化）
    #     conv2 = Conv2D(filters=256, kernel_size=(2, 11), strides=1,  padding='same')(conv1_pool)
    #     conv2 = keras.layers.PReLU(shared_axes=[1, 2])(conv2)
    #     conv2_pool = AbMaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    #     # conv block -3 （卷积）
    #     conv3 = Conv2D(filters=128, kernel_size=(2, 21), strides=1, padding='same')(conv2_pool)
    #     encoder = keras.layers.PReLU(shared_axes=[1, 2])(conv3)
    #     # encoder = AbMaxPooling2D(pool_size=())(conv3) 无池化操作
    #
    #     # decoder
    #     # conv block -1 （反卷积+反池化）
    #     deconv1 = Conv2DTranspose(filters=128, kernel_size=(2, 21), strides=1, padding='same')(encoder)
    #     deconv1 = keras.layers.PReLU(shared_axes=[1, 2])(deconv1)
    #     deconv1_unpool = gen_nn_ops.max_pool_grad(conv2,  # 池化前的tensor，即max pool的输入
    #                                               conv2_pool,  # 池化后的tensor，即max pool 的输出
    #                                               deconv1,  # 需要进行反池化操作的tensor
    #                                               ksize=[1, 2, 2, 1],
    #                                               strides=[1, 2, 2, 1],
    #                                               padding='SAME')
    #     # conv block -2 （反卷积+池化）
    #     deconv2 = Conv2DTranspose(filters=256, kernel_size=(2, 11), strides=1, padding='same')(deconv1_unpool)
    #     deconv2 = keras.layers.PReLU(shared_axes=[1, 2])(deconv2)
    #     deconv2_unpool = gen_nn_ops.max_pool_grad(conv1,  # 池化前的tensor，即max pool的输入
    #                                               conv1_pool,  # 池化后的tensor，即max pool 的输出
    #                                               deconv2,  # 需要进行反池化操作的tensor
    #                                               ksize=[1, 2, 2, 1],
    #                                               strides=[1, 2, 2, 1],
    #                                               padding='SAME')
    #     # conv block -3 （反卷积）
    #     deconv3 = Conv2DTranspose(filters=512, kernel_size=(2, 5), strides=1, padding='same')(deconv2_unpool)
    #     decoder = keras.layers.PReLU(shared_axes=[1, 2])(deconv3)
    #     output_layer = decoder
    #
    #     model = Model(inputs=input_layer, outputs=output_layer)
    #
    #     # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    #     model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.00001), metrics=['accuracy'])
    #
    #     file_path = self.output_directory + 'best_model.hdf5'
    #
    #     #
    #     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
    #                                                        monitor='loss', save_best_only=True)
    #
    #     self.callbacks = [model_checkpoint]
    #
    #     return model
    #
    # def fit(self, x_train, y_train, x_val, y_val, y_true):
    #     # x_val and y_val are only used to monitor the test loss and NOT for training
    #     batch_size = 12
    #     nb_epochs = 100
    #
    #     # 小批量训练大小
    #     mini_batch_size = batch_size
    #
    #     start_time = time.time()
    #
    #     # 训练模型
    #     hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
    #                           verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
    #
    #     duration = time.time() - start_time
    #
    #     model = keras.models.load_model(self.output_directory + 'best_model.hdf5')
    #
    #     y_pred = model.predict(x_val)
    #
    #     # convert the predicted from binary to integer
    #     y_pred = np.argmax(y_pred, axis=1)
    #
    #     # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)
    #
    #     keras.backend.clear_session()
