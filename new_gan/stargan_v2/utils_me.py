import tensorflow as tf
import numpy as np
import cv2

weight_init = tf.contrib.layers.variance_scaling_initializer()

def resblock(input_x,channel,kernel_size,strides,padding='SAME',scope_name=''):
    with tf.variable_scope(scope_name):
        _, _, _, init_channel = input_x.get_shape().as_list()
        
        x = tf.nn.relu(input_x)
        x = tf.contrib.layers.instance_norm(x,epsilon=1e-05,center=True, scale=True)
        x = tf.layers.conv2d(x,channel,kernel_size,padding=padding,kernel_initializer=weight_init)
        
        x = tf.nn.relu(x)
        x = tf.contrib.layers.instance_norm(x,epsilon=1e-05,center=True, scale=True)
        x = tf.layers.conv2d(x,channel,kernel_size,padding=padding,kernel_initializer=weight_init)
        
        if(init_channel==channel):
            return input_x+x
        else:
            input_x = tf.layers.conv2d(input_x,channel,1,1,padding=padding)
            return x+input_x

def resblock_no_norm(input_x,channel,kernel_size,strides,padding='SAME',scope_name=''):
    with tf.variable_scope(scope_name):
        _, _, _, init_channel = input_x.get_shape().as_list()
        
        x = tf.nn.leaky_relu(input_x,0.2)
        x = tf.layers.conv2d(x,channel,kernel_size,padding=padding,kernel_initializer=weight_init)
        
        x = tf.nn.leaky_relu(x,0.2)
        x = tf.layers.conv2d(x,channel,kernel_size,padding=padding,kernel_initializer=weight_init)
        
        if(init_channel==channel):
            return input_x+x
        else:
            input_x = tf.layers.conv2d(input_x,channel,1,1,padding=padding)
            return x+input_x

def resblock_adain(input_x,channel,kernel_size,strides,gamma1,beta1,gamma2,beta2,padding='SAME',scope_name=''):
    with tf.variable_scope(scope_name):
        _, _, _, init_channel = input_x.get_shape().as_list()
        
        x = tf.nn.relu(input_x)
        x = adaptive_instance_norm(x,gamma1,beta1)
        x = tf.layers.conv2d(x,channel,kernel_size,padding=padding,kernel_initializer=weight_init)
        
        x = tf.nn.relu(x)
        x = adaptive_instance_norm(x,gamma2,beta2)
        x = tf.layers.conv2d(x,channel,kernel_size,padding=padding,kernel_initializer=weight_init)
        
        if(init_channel==channel):
            return input_x+x
        else:
            input_x = tf.layers.conv2d(input_x,channel,1,1,padding=padding)
            return x+input_x
            

def adaptive_instance_norm(original,gamma1,beta1,epsilon=1e-5):
    c_mean, c_var = tf.nn.moments(original, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)
    return gamma1 * ((original - c_mean) / c_std) + beta1
    