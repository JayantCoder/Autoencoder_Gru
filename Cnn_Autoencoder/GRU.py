import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time



class Single_GRU:  # 所有时间步一个时间窗的gru
    """
        model初始化
    """
    def __init__(self, dataset_name, input_shape, output_shape, verbose=True):
        self.dataset_name = dataset_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.verbose = verbose
        self.now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model = self.build_model()  # 调用 build_model
        if(verbose==True):
            self.model.summary()  # 打印 model 结构

    """
        创建model
    """
    def build_model(self):
        # model 网络结构
        input_layer = tf.keras.Input(self.input_shape)
        layer_1 = tf.keras.layers.GRU(20, return_sequences=True)(input_layer)
        layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
        layer_3 = tf.keras.layers.Activation(activation='tanh', name='gru_output')(layer_2)
        output_layer = tf.keras.layers.Dense(self.output_shape)(layer_3)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        # 编译model
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        # 设置 callbacks 回调
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',  # 用于动态调整learning rate
                                                         factor=0.5,
                                                         patience=50,
                                                         min_lr=0.0001)

        self.model_path = 'result' + '/gru/' + self.dataset_name + self.now_time + '_gru_model.hdf5'

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                              monitor='loss',
                                                              save_best_only=True)
        log_dir = "logs/gru/" + self.now_time
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.callbacks = [reduce_lr, model_checkpoint]

        return model

        # 训练model

    def fit(self, x_train, y_train, x_val, y_val, epochs=500):
        """
            验证集: (x_val, y_val) 用于监控loss，防止overfitting
        """
        batch_size = 6
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train,
                              validation_data=(x_val, y_val),
                              batch_size=mini_batch_size,
                              epochs=epochs,
                              verbose=self.verbose,
                              callbacks=self.callbacks)

        duration = time.time() - start_time

        tf.keras.backend.clear_session()  # 清除当前tf计算图


class Multi_Gru:  # 不同时间窗的gru
    """
        model初始化
    """
    def __init__(self, dataset_name, input_shape, output_shape=1, h_dim=1, timewindows=1, verbose=True):
        self.dataset_name = dataset_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.h_dim = h_dim
        self.timewindows = timewindows
        self.gru_num = int(self.input_shape[0] / self.timewindows)  # Gru数量
        self.depth = 3
        self.now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # 动态创建layers变量名
        self.layers = self.__dict__
        for width in range(self.gru_num):
            self.layers['gru_' + str(width) + '_input_layer'] = tf.zeros([self.timewindows, self.input_shape[1]])
            for depth in range(1, self.depth + 1):
                self.layers['gru_' + str(width) + '_layer_' + str(depth)] = tf.zeros([self.timewindows, self.h_dim])

        # 动态创建多输入x_train_{index} 和 x_val_{index}用于 方法fit
        self.multi_train = self.__dict__
        self.multi_val = self.__dict__
        self.multi_test = self.__dict__
        for index in range(self.gru_num):
            self.multi_train['x_train_' + str(index)] = tf.zeros([1, self.timewindows, self.input_shape[1]])
            self.multi_val['x_val_' + str(index)] = tf.zeros([1, 1, 1])
            self.multi_test['x_test_' + str(index)] = tf.zeros([1, 1, 1])
        self.callbacks = []
        self.verbose = verbose

        self.model = self.build_model()  # 调用 build_model
        if (verbose == True):
            self.model.summary()  # 打印 model 结构
        # print(self.__dict__.keys())
    """
        创建model
    """

    def build_model(self):
        # model 网络结构
        # input_layer = tf.keras.Input(self.input_shape)  # 输入 (None, 568, 1)
        for width in range(self.gru_num):
            self.layers['gru_' + str(width) + '_input_layer'] = tf.keras.Input((self.timewindows, self.input_shape[1]),
                                                                               name=f'gru_input_{width}')
            self.layers['gru_' + str(width) + '_layer_1'] = tf.keras.layers.GRU(self.h_dim, return_sequences=True)(
                self.layers['gru_' + str(width) + '_input_layer'])
            # layer_1 = tf.keras.layers.GRU(20, return_sequences=True)(input_layer)
            self.layers['gru_' + str(width) + '_layer_2'] = tf.keras.layers.BatchNormalization()(
                self.layers['gru_' + str(width) + '_layer_1'])
            # layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
            self.layers['gru_' + str(width) + '_layer_3'] = tf.keras.layers.Activation(activation='tanh',
                                                                                       name=f'gru_output_{width}')(
                self.layers['gru_' + str(width) + '_layer_2'])
            # layer_3 = tf.keras.layers.Activation(activation='tanh', name='gru_output')(layer_2) self.layers['gru_'
            # + str(width) + '_layer_4'] = tf.keras.layers.Dense(self.output_shape)(self.layers['gru_' + str(width) +
            # '_layer_3']) output_layer = tf.keras.layers.Dense(self.output_shape)(layer_3)
        finalGruList = [self.layers['gru_' + str(width) + '_layer_3'] for width in range(self.gru_num)]
        concat_layer = tf.keras.layers.concatenate(finalGruList, axis=1, name='concat_layer')
        output_layer = tf.keras.layers.Dense(self.output_shape, name='dense_layer')(concat_layer)

        # 创建model object
        model = tf.keras.Model(
            inputs=[self.layers['gru_' + str(width) + '_input_layer'] for width in range(self.gru_num)],
            outputs=output_layer)

        # 编译model
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        # 设置 callbacks 回调
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',  # 用于动态调整learning rate
                                                         factor=0.5,
                                                         patience=50,
                                                         min_lr=0.0001)

        self.model_path = 'result' + '/multigru/' + self.dataset_name + self.now_time + '_gru_model.hdf5'

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                              monitor='loss',
                                                              save_best_only=True)
        log_dir = "logs/gru/" + self.now_time
        tboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.callbacks = [reduce_lr, model_checkpoint, tboard]

        return model

        # 训练model

    def fit(self, x_train, y_train, x_val, y_val, epochs=500):
        """
            验证集: (x_val, y_val) 用于监控loss，防止overfitting
        """
        batch_size = 16
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        for i in range(self.gru_num):
            self.multi_train['x_train_' + str(i)] = x_train[:, (i * self.timewindows):((i + 1) * self.timewindows), :]
            self.multi_val['x_val_' + str(i)] = x_val[:, (i * self.timewindows):((i + 1) * self.timewindows), :]

        x_train_list = [self.multi_train['x_train_' + str(i)] for i in range(self.gru_num)]
        x_val_list = [self.multi_val['x_val_' + str(i)] for i in range(self.gru_num)]
        hist = self.model.fit(x_train_list, y_train,
                              validation_data=(x_val_list, y_val),
                              batch_size=mini_batch_size,
                              epochs=epochs,
                              verbose=self.verbose,
                              callbacks=self.callbacks)

        tf.keras.backend.clear_session()  # 清除当前tf计算图


def output_of_multi_gru(x_test, model_path, timewindows=1):  # 提取Gru的输出
    '''
    @train_data:模型的输入，例如Car_TRAIN.txt
    @filepath:训练好的模型参数
    '''
    input_shape = x_test.shape[1:]
    gru_num = int(input_shape[0] / timewindows)
    multi_test = dict()
    for i in range(gru_num):
        multi_test['x_test_' + str(i)] = x_test[:, (i * timewindows):((i + 1) * timewindows), :]

    x_test_list = [multi_test['x_test_' + str(i)] for i in range(gru_num)]

    initial_model = tf.keras.models.load_model(model_path)

    gru_output_model = tf.keras.Model(inputs=initial_model.input,
                                      outputs=initial_model.get_layer('concat_layer').output)

    output = gru_output_model.predict(x_test_list)
    output = np.expand_dims(output, 1)
    return np.array(output[:, :, ::timewindows, :])


def output_of_single_gru(train_data, model_path, timewindows=1):  # 提取Gru的输出
    '''
    @train_data:模型的输入，例如Car_TRAIN.txt
    @filepath:训练好的模型参数
    @timewindows:时间窗
    '''
    initial_model = tf.keras.models.load_model(model_path)

    gru_output_model = tf.keras.Model(inputs=initial_model.input,
                                      outputs=initial_model.get_layer('gru_output').output)

    output = gru_output_model.predict(train_data)
    output = np.expand_dims(output, 1)
    return np.array(output[:, ::timewindows])