import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Autoencoder_GRU:  # GRU神经网络
    """
        model初始化
    """
    def __init__(self, output_directory, input_shape, output_shape, verbose=True):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.verbose = verbose

        self.model = self.build_model()  # 调用 build_model
        if (verbose == True):
            self.model.summary()  # 打印 model 结构
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        init_model_file = os.path.join(self.output_directory, 'gru/model_init.hdf5')
        self.model.save_weights(init_model_file)  # 保存初始权重

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

        file_path = os.path.join(self.output_directory, 'gru/Car_gru_model.hdf5')

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                              monitor='loss',
                                                              save_best_only=True)
        log_dir = "logs/gru/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.callbacks = [reduce_lr, model_checkpoint]

        return model

        # 训练model

    def fit(self, x_train, y_train, x_val, y_val, epochs=500):
        """
            验证集: (x_val, y_val) 用于监控loss，防止overfitting
        """
        batch_size = 16
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


def output_of_gru(train_data, filepath, timewindows=1):  # 提取Gru的输出
    '''
    @train_data:模型的输入，例如Car_TRAIN.txt
    @filepath:训练好的模型参数
    @timewindows:时间窗
    '''
    initial_model = tf.keras.models.load_model(filepath)

    gru_output_model = tf.keras.Model(inputs=initial_model.input,
                                      outputs=initial_model.get_layer('gru_output').output)

    output = gru_output_model.predict(train_data)
    output = np.expand_dims(output, 1)
    return np.array(output[:, ::timewindows])