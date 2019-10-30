import tensorflow as tf
import numpy as np


def blur_pooling(x,filt_size,stride=2,pool_type='avg',scope_name='',trainable=None):
    with tf.variable_scope(scope_name):
        if(pool_type=='avg'):
            x = x
        elif(pool_type=='max'):
            x = tf.layers.max_pooling2d(x,2,1,padding='VALID')
        elif(pool_type=='conv'):
            x = tf.layers.conv2d(x,x.shape[-1],3,1,padding='SAME')

        if(filt_size==3):
            a = np.array([1., 2., 1.])
        elif(filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        pad_off = 0
        pad_sizes = (filt_size-1)//2
        off = int((stride-1)/2.)
        channels = x.shape[-1]
        filt = a[:,None]*a[None,:]
        filt = filt/np.sum(filt)
        filt = filt[:,:,None,None]
        fi_ = np.tile(filt,[1,1,channels,1])
        w = tf.get_variable(name='anti_filter',dtype=tf.float32,shape=fi_.shape,trainable=False,initializer=tf.constant_initializer(fi_))
        w = tf.identity(w)
        x_ = x = tf.pad(x,[[0,0],[pad_sizes,pad_sizes],
                               [pad_sizes,pad_sizes],[0,0]])
        x_ = tf.nn.depthwise_conv2d(x_,w,(1,2,2,1),padding='VALID')
        return x_

def upsample(x):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    return x

def first_oct_conv(x,ch_out,kernel_size,stride,padding='SAME',alpha=0.25,scope_name=''):
    with tf.variable_scope(scope_name):
        l_ch_out = int(ch_out*(1-alpha))
        h_ch_out = ch_out-l_ch_out

        x2h = tf.layers.conv2d(x,h_ch_out,kernel_size,stride,padding=padding,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        
        x2l = blur_pooling(x,5)
        x2l = tf.layers.conv2d(x2l,l_ch_out,kernel_size,stride,padding=padding,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

        return x2h,x2l

def normal_oct_conv(h_conv,l_conv,ch_out,kernel_size,stride,padding='SAME',alpha=0.25,scope_name=''):
    with tf.variable_scope(scope_name):
        if(alpha!=0):
            l_ch_out = int(ch_out*(1-alpha))
            h_ch_out = ch_out-l_ch_out
        else:
            l_ch_out = l_conv.shape[-1]
            h_ch_out = h_conv.shape[-1]

        if(stride==2):
            h_conv = blur_pooling(h_conv,5,scope_name='blur0')
            l_conv = blur_pooling(l_conv,5,scope_name='blur1')


        h2h =  tf.layers.conv2d(h_conv,h_ch_out,kernel_size,1,padding=padding,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

        h2l = blur_pooling(h_conv,5,scope_name='blur2')
        h2l = tf.layers.conv2d(h2l,l_ch_out,kernel_size,1,padding=padding,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

        l2l = tf.layers.conv2d(l_conv,l_ch_out,kernel_size,1,padding=padding,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        
        l2h = tf.layers.conv2d(l_conv,h_ch_out,kernel_size,1,padding=padding,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        l2h = upsample(l2h)
        
        h_conv = h2h+l2h
        l_conv = l2l+h2l
        return h_conv,l_conv

def last_oct_conv(h_conv,l_conv,ch_out,kernel_size,stride,padding='SAME',alpha=0.25,scope_name=''):
    with tf.variable_scope(scope_name):

        if(stride==2):
            h_conv = tf.layers.average_pooling2d(h_conv,2,2,padding='SAME')
            l_conv = tf.layers.average_pooling2d(l_conv,2,2,padding='SAME')

        h_conv = tf.layers.conv2d(h_conv,ch_out,kernel_size,1,padding=padding,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        l_conv = tf.layers.conv2d(l_conv,ch_out,kernel_size,1,padding=padding,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

        l_conv = upsample(l_conv)
        return h_conv+l_conv
