import tensorflow as tf
import numpy as np



def upsample(x):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    return x

def first_oct_conv(x,ch_out,kernel_size,stride,padding='SAME',alpha=0.25):
    ch_out = int(ch_out*(1-alpha))
    input_x = x
    
    h_conv = tf.layers.conv2d(input_x,ch_out,kernel_size,stride,padding=padding)
    l_conv = tf.layers.average_pooling2d(x,2,2,padding='SAME')
    l_conv = tf.layers.conv2d(l_conv,ch_out,kernel_size,stride,padding=padding)
    
    return h_conv,l_conv

def normal_oct_conv(h_conv,l_conv,ch_out,kernel_size,stride,padding='SAME',alpha=0.25):
    ch_out = int(ch_out*(1-alpha))
    
    if(stride==2):
        h_conv = tf.layers.average_pooling2d(h_conv,2,2,padding='SAME')
        l_conv_up = l_conv
        l_conv = tf.layers.average_pooling2d(l_conv,2,2,padding='SAME')
    else:
        l_conv_up = upsample(l_conv)
        
    h_conv =  tf.layers.conv2d(h_conv,ch_out,kernel_size,1,padding=padding)
    h_conv_pool = tf.layers.average_pooling2d(h_conv,2,2,padding='SAME')
    h_conv_pool_conv = tf.layers.conv2d(h_conv_pool,ch_out,kernel_size,1,padding=padding)
    
    l_conv = tf.layers.conv2d(l_conv,ch_out,kernel_size,1,padding=padding)
    
    h_conv = h_conv + l_conv_up
    l_conv = l_conv_up + h_conv_pool_conv
    return h_conv,l_conv

def lsat_oct_conv(h_conv,l_conv,ch_out,kernel_size,stride,padding='SAME',alpha=0.25):
    ch_out = int(ch_out*(1-alpha))
    
    if(stride==2):
        h_conv = tf.layers.average_pooling2d(h_conv,2,2,padding='SAME')
        l_conv = tf.layers.average_pooling2d(l_conv,2,2,padding='SAME')
        
    h_conv = tf.layers.conv2d(h_conv,ch_out,kernel_size,1,padding=padding)
    l_conv = tf.layers.conv2d(l_conv,ch_out,kernel_size,1,padding=padding)
    
    h_conv = upsample(h_conv)
    
    return h_conv+l_conv
    