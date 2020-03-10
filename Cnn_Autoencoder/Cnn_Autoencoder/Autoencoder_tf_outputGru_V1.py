import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time 

def readucr(filename):#读取数据
    data = np.loadtxt(filename, dtype=str, delimiter=',') 
    x = np.array(data[:, 0:574], dtype=np.float32)
    y = np.array(data[:, 1:575], dtype=np.float32)
    x = x.reshape(x.shape[0], -1, 1)
    y = y.reshape(y.shape[0], -1, 1)
    print(f'{filename}: {x.shape}, {y.shape}')
    return x, y

class Autoencoder_GRU: # GRU神经网络
    """
        model初始化
    """
    def __init__(self, output_directory, input_shape, output_shape, verbose= True): 
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.verbose = verbose
        
        self.model = self.build_model() # 调用 build_model
        if(verbose == True):
            self.model.summary() # 打印 model 结构
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        init_model_file = os.path.join(self.output_directory, 'model_init.hdf5')
        self.model.save_weights(init_model_file) # 保存初始权重

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
        model.compile(loss='mse', optimizer='adam',metrics=['mae'])
        
        # 设置 callbacks 回调
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', # 用于动态调整learning rate
                                                         factor=0.5, 
                                                         patience=50, 
                                                         min_lr=0.0001)

        file_path =  os.path.join(self.output_directory, 'best1_model.hdf5')

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                              monitor='loss', 
                                                              save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model 

    # 训练model
    def fit(self, x_train, y_train, x_val, y_val, epochs): 
        """
            验证集: (x_val, y_val) 用于监控loss，防止overfitting
        """  
        batch_size = 16
        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, 
                              validation_data=(x_val, y_val),
                              batch_size=mini_batch_size, 
                              epochs=epochs,
                              verbose=self.verbose,  
                              callbacks=self.callbacks)
        
        duration = time.time() - start_time

        tf.keras.backend.clear_session() # 清除当前tf计算图



if __name__ == '__main__':
    X_train,Y_train = readucr('data/Car_TRAIN.txt')
    X_test,Y_test = readucr('data/Car_TEST.txt')

    input_shape = X_train.shape[1:]
    model = Autoencoder_GRU('test', input_shape, 1)
    model.fit(X_train, Y_train, X_test, Y_test, 10)
    initial_model = tf.keras.models.load_model('test/best1_model.hdf5')

    gru_output_model = tf.keras.Model(
                                        inputs = initial_model.input,
                                        outputs = initial_model.get_layer('gru_output').output
                                    )

    pred = gru_output_model.predict(X_test)
    print(pred)
    print(pred.shape)