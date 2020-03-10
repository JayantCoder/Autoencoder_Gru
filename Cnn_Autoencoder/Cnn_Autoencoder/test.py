import tensorflow as tf
from tensorflow import keras

z_dim = 10
class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = keras.layers.Dense(128)
        # mean
        self.fc2 = keras.layers.Dense(z_dim)
        # variance
        self.fc3 = keras.layers.Dense(z_dim)

        self.fc4 = keras.layers.Dense(128)
        self.fc5 = keras.layers.Dense(784)

    def encoder(self, x):
        pass

    def decoder(self, out):
        pass

    def call(self):
        pass