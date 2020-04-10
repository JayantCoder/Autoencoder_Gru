import tensorflow as tf
import numpy as np
from tensorflow import keras
#
# batchsz = 100
# z_dim = 10
# class VAE(keras.Model):
#     def __init__(self):
#         super(VAE, self).__init__()
#
#         # Encoder
#         self.fc1 = keras.layers.Dense(128)
#         # mean
#         self.fc2 = keras.layers.Dense(z_dim)
#         # variance
#         self.fc3 = keras.layers.Dense(z_dim)
#
#         # Decoder
#         self.fc4 = keras.layers.Dense(128)
#         self.fc5 = keras.layers.Dense(784)
#
#     def encoder(self, x):
#         h = tf.nn.relu(self.fc1(x))
#
#         mu = self.fc2(h)
#
#         log_var = self.fc3(h)
#         return  mu, log_var
#
#     def decoder(self, z):
#         out = tf.nn.relu(self.fc4(z))
#         out = self.fc5(out)
#
#         return out
#
#     def reparameterize(self, mu, log_var):
#         # 服从正态分布的eps
#         eps = tf.random.normal(log_var.shape)
#         std = tf.exp(log_var) ** 0.5
#         z = mu + std * eps
#         return z
#
#     def call(self, inputs, training=None):
#         mu, log_var = self.encoder(inputs)
#
#         z = self.reparameterize(mu, log_var)
#
#         outputs = self.decoder(z)
#         return outputs, mu, log_var
#
# train_dataset = tf.data.Dataset.from_tensor_slices()
#
# model = VAE()
# model.build(input_shape=(None, 784))
# for epochs in range(1000):
#
#     for step, x in enumerate(train_dataset):
#
#         with tf.GradientTape as tape:
#             x_rec_logits, mu, log_var = model(x)
#             rec_loss = tf.sigmoid(x, x_rec_logits, from_logits=True)
#             rec_loss = tf.reduce_mean(rec_loss) / x.shape[0]
#
#             kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
#             kl_div = tf.reduce_mean / x.shape[0]
#
#             loss = rec_loss + 1. * kl_div
#
#         grads = tape.gradient(loss, model.trainable_variables)
#         keras.optimizers.apply_gradients(zip(grads, model.trainable_variables))
#
#
#     # evaluation 生成图片
#     z = tf.random.normal((batchsz, z_dim))
#     logits = model.decoder(z)
#     x_hat = tf.sigmoid(logits)
#     x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255
#     x_hat = x_hat.astype(np.uint8)
# def unAbMaxPooling(inputs_argmax, ksize, strides=2, padding='SAME'):
#     # 假定 ksize = strides
#     inputs = inputs_argmax[0]
#     argmax = inputs_argmax[1]
#     input_shape = inputs.get_shape()
#     if padding == 'SAME':
#         rows = input_shape[1] * ksize[1]
#         cols = input_shape[2] * ksize[2]
#     else:
#         rows = (input_shape[1] - 1) * ksize[1] + ksize[1]
#         cols = (input_shape[2] - 1) * ksize[2] + ksize[2]
#     # 计算new shape
#     output_shape = (input_shape[0], rows, cols, input_shape[3])
#     # 计算索引
#     one_like_mask = tf.ones_like(argmax)
#     batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
#     b = one_like_mask * batch_range
#     y = argmax // (output_shape[2] * output_shape[3])
#     x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
#     feature_range = tf.range(output_shape[3], dtype=tf.int64)
#     c = one_like_mask * feature_range
#     # 转置索引
#     update_size = tf.size(inputs)
#     indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, update_size]))
#     values = tf.reshape(inputs, [update_size])
#     outputs = tf.scatter_nd(indices, values, output_shape)
#     return outputs
#
# inputs = tf.random.normal((2, 2, 2, 2))
from cnn_AE_2 import Cnn_AE_2
from cnn_AE_n import Cnn_AE_n
from cnn_AE_npron import Cnn_AE_npron

model = Cnn_AE_n('result', (4, 512, 20), 6, True)
# model = Cnn_AE_npron('result', (4, 512, 20), 6, True)