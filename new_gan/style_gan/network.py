import tensorflow as tf
import numpy as np

he_normal = tf.contrib.layers.variance_scaling_initializer()

level = 5


ch_ = [512,256,128,64,32,16]

def discriminator(x,stage,alpha):
    
    with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
        print('discriminator------')
        
        if(stage!=1):
            with tf.variable_scope('from_RGB'+str(stage-1)):
                #ch = 16*np.power(2,level-stage)
                ch = ch_[stage]
                s_x = tf.layers.conv2d(x,ch*2,1,1,padding='SAME',kernel_initializer=he_normal)
                s_x = tf.nn.leaky_relu(s_x)
                s_x = tf.layers.average_pooling2d(s_x,2,2,padding='SAME')
        
        with tf.variable_scope('from_RGB'+str(stage)):
            #ch = 16*np.power(2,level-stage)
            ch = ch_[stage]
            x = tf.layers.conv2d(x,ch,1,1,kernel_initializer=he_normal)
            x = tf.nn.leaky_relu(x)
        
        for i in range(1,stage):
            with tf.variable_scope(str(level-stage+i)):
                #ch = 16*np.power(2,level-stage+i-1)
                ch = ch_[-(level-stage+i)]
                x = tf.layers.conv2d(x,ch,3,1,padding='SAME',kernel_initializer=he_normal)
                x = tf.nn.leaky_relu(x)
                
                x = tf.layers.conv2d(x,ch*2,3,1,padding='SAME',kernel_initializer=he_normal)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                if(i==1):
                    x = x*alpha+s_x*(1-alpha)
            print(x)
        with tf.variable_scope('scale'+str(stage)):
            x = tf.layers.conv2d(x,ch_[0],3,1,padding='SAME',kernel_initializer=he_normal)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x,ch_[0],4,1,padding='VALID',kernel_initializer=he_normal)
            x = tf.nn.leaky_relu(x)
            #x = MinibatchstateConcat(x)
            x = tf.layers.flatten(x)
            x_logit = tf.layers.dense(x,1)
        return x_logit

def generator(z,stage,alpha):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        print('generator---------')
        z = Pixel_Norm(z)
        z = tf.layers.dense(z,4*4*ch_[0])
        z = tf.reshape(z,[-1,4,4,ch_[0]])
        z = Pixel_Norm(tf.nn.leaky_relu(z))
        z = tf.layers.conv2d(z,ch_[0],3,1,padding='SAME',kernel_initializer=he_normal)
        z = Pixel_Norm(tf.nn.leaky_relu(z))
        
        for i in range(1,stage):
            z = upsample(z)
            if(i==stage-1):
                with tf.variable_scope('to_RGB'+str(stage-1)):
                    s_z = tf.layers.conv2d(z,3,1,1,kernel_initializer=he_normal)
            
            #ch = 16*np.power(2,level-1-i)
            ch = ch_[i]
            z = tf.layers.conv2d(z,ch,3,1,padding='SAME',kernel_initializer=he_normal)
            z = Pixel_Norm(tf.nn.leaky_relu(z))
            z = tf.layers.conv2d(z,ch,3,1,padding='SAME',kernel_initializer=he_normal)
            z = Pixel_Norm(tf.nn.leaky_relu(z))
            print(z)
        with tf.variable_scope('to_RGB'+str(stage)):
            z = tf.layers.conv2d(z,3,1,1,kernel_initializer=he_normal)
        
        if(stage==1):
            return z
        
        return z*alpha+s_z*(1-alpha)
        
        
def upsample(x):
    x = tf.image.resize_nearest_neighbor(x,[x.shape[1]*2,x.shape[2]*2])
    return x

def MinibatchstateConcat(input, averaging='all'):
    #s = input.shape
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")

    vals = tf.tile(vals, multiples=[input.shape[0], input.shape[1], input.shape[2], 1])
    return tf.concat([input, vals], axis=3)
        
        
def Pixel_Norm(x, eps=1e-8):
    if len(x.shape) > 2:
        axis_ = 3
    else:
        axis_ = 1
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keep_dims=True) + eps)
