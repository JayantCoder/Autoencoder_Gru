# Import all the required Libraries
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, \
    Input, UpSampling2D, ZeroPadding1D, ZeroPadding2D, Lambda, Conv2DTranspose


# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def abMaxPooling1D(inputs, pool_size=2, strides=2, padding='SAME'):
    # tf.nn.max_pool(value, ksize, strides, padding, name=None)
    # output1 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
    # output2 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
    output1 = tf.nn.max_pool1d(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2 = tf.nn.max_pool1d(-inputs, ksize=pool_size, strides=strides, padding=padding)
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


class Cnn_AE_1:
    def __init__(self, output_directory, input_shape, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape)
        # verbose是信息展示模式
        if verbose == True:
            self.model.summary()
        self.verbose = verbose

    def build_model(self, input_shape):
        # input --> (None, 1, 576, 1)
        input_layer = Input(shape=input_shape)
        # Encoder
        # conv block -1 （卷积+池化）
        # 每个feature map是一个一维张量
        if len(input_layer.shape) == 3:
            # conv1 = ZeroPadding1D(1)(input_layer)
            conv1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
            # conv1 = Conv2D(filters=16, kernel_size=(input_layer.shape[1], 5), activation='relu', padding='same')(
            #         input_layer)
            conv1_pool = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': 2})(conv1)
            # conv1_pool = AbMaxPooling1D(pool_size=2)(conv1)
            # conv1_pool = tf.expand_dims(tf.transpose(conv1_pool, [0, 2, 1]), -1)
            # conv1_pool = Lambda(reshapes, arguments={'retype': 'reshapedim'})(conv1_pool)
        if len(input_layer.shape) == 4:
            conv1 = ZeroPadding2D((0, 1))(input_layer)
            h1 = input_layer.shape[1]
            conv1 = Conv2D(filters=16, kernel_size=(h1, 3), activation='relu')(conv1)
            # conv1 = Lambda(reshapes, arguments={'retype': 'squeeze'})(conv1)
            conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': 2}, name='abMaxPool1')(conv1)
            conv1_pool = Lambda(reshapes, name='reshape1')(conv1_pool)

        # conv block -2 （卷积+池化）
        conv2 = ZeroPadding2D((0, 1))(conv1_pool)
        h2 = conv2.shape[1]
        conv2 = Conv2D(filters=8, kernel_size=(h2, 3), activation='relu')(conv2)
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': 2}, name='abMaxPool2')(conv2)
        conv2_pool = Lambda(reshapes, name='reshape2')(conv2_pool)

        # conv block -3 （卷积）
        conv3 = ZeroPadding2D((0, 1))(conv2_pool)
        h3 = conv3.shape[1]
        conv3 = Conv2D(filters=8, kernel_size=(h3, 3), activation='relu')(conv3)
        # encoder = keras.layers.PReLU(shared_axes=[1, 2])(conv3)
        encoder, conv3_argmax  = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': 2}, name='abMaxPool3')(conv3)

        # decoder
        # conv block -1 （反卷积+反池化）
        # deconv1_unpool = UpSampling2D(size=(1, 2))(encoder)
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'argmax': conv3_argmax, 'strides': 2}, name='unAbPool1')(encoder)
        deconv1 = Conv2DTranspose(filters=8, kernel_size=(h3, 3), padding='same', activation='relu')(deconv1_unpool)
        # deconv1 = ZeroPadding2D((0, 1))(deconv1_unpool)
        # deconv1 = Conv2D(filters=8, kernel_size=(h3, 3), activation='relu')(deconv1)

        # conv block -2 （反卷积+反池化）
        # deconv2_unpool = UpSampling2D(size=(1, 2))(deconv1)
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'argmax': conv2_argmax, 'strides': 2}, name='unAbPool2')(deconv1)
        deconv2 = Conv2DTranspose(filters=8, kernel_size=(h2, 3), padding='same', activation='relu')(deconv2_unpool)
        # deconv2 = ZeroPadding2D((0, 1))(deconv2_unpool)
        # deconv2 = Conv2DTranspose(filters=8, kernel_size=(h2, 3), padding='same', activation='relu')(deconv2)

        # conv block -3 （反卷积+反池化）
        # deconv3_unpool = UpSampling2D(size=2)(deconv2)
        deconv3_unpool = Lambda(unAbMaxPooling, arguments={'argmax': conv1_argmax, 'strides': 2}, name='unAbPool3')(deconv2)
        deconv3 = Conv2DTranspose(filters=16, kernel_size=(h1, 3), padding='same', activation='relu')(deconv3_unpool)


        decoder = ZeroPadding2D((0, 1))(deconv3)
        output_layer = Conv2D(filters=1, kernel_size=(decoder.shape[1], 3), activation='sigmoid')(decoder)

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

        # 开始时间
        start_time = time.time()

        file_path = os.path.join(self.output_directory, 'best_model.hdf5')

        # 训练模型
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                       verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        # print('history：', hist.history)
        # 训练持续时间
        duration = time.time() - start_time

        # self.model.save(file_path)
        print('duration: ', duration)

        # 做测试，所以需要加载模型
        model = load_model(file_path, custom_objects={'abMaxPooling_with_argmax': abMaxPooling_with_argmax})

        loss = model.evaluate(x_val, y_val, batch_size=mini_batch_size, verbose=0)
        # y_pred = model.predict(x_val)
        # y_pred = np.argmax(y_pred, axis=1)
        print('test_loss: ', loss)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()
