import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2

batch_size = 64
n_dim = 100
stride = 5
noise_num = 100
lr = 0.0001
mnist = input_data.read_data_sets('./MNIST', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def batch_leaky(x,trainable=True):
    x = tf.contrib.layers.batch_norm(x,is_training=trainable)
    x = tf.nn.leaky_relu(x)
    return x

def decode(image,trainable):
    pass

def encode():
    pass

def discriminator(image,reuse=False,trainable=True):
    with tf.variable_scope('discriminator',reuse=reuse):
        pass

def generator(z,reuse=False,trainable=True):
    with tf.variable_scope('generator',reuse=reuse):
        pass

def check_enviroment():
    if not os.path.exists('output/'):
        os.makedirs('output/')

def noise_sample(batch_size,noise_dim):
    return np.random.uniform(-1., 1., size=[batch_size, noise_dim])

def get_img():
    mnist_img, _ = mnist.train.next_batch(batch_size)
    mnist_img = np.reshape(mnist_img,(-1,28,28,1))
    return mnist_img

if(__name__=='__main__'):
    pass