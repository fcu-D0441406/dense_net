import tensorflow as tf
import numpy as np

bk = [2,2,2,2]

class Resnet_v2:
    
    def __init__(self,img_size,channel,class_num,batch_size=64,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.img_size,self.img_size,self.channel])
        self.weight_init = tf.contrib.layers.variance_scaling_initializer()
        self.build_net(trainable)
 
        
    
    def build_net(self,trainable):
        with tf.variable_scope('resnet_v2'):
            #tf.contrib.layers.variance_scaling_initializer()
            net = tf.layers.conv2d(self.x,64,7,2,padding='SAME',
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                                
            net = tf.layers.batch_normalization(net,training=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net,3,2,padding='SAME')#128
            print(net)
            resnet1 = self.res_block(net,64,1,bk[0],trainable,'res_block0')
            print(resnet1)
            
            resnet2 = self.res_block(resnet1,128,2,bk[1],trainable,'res_block1')
            print(resnet2)
            
            resnet3 = self.res_block(resnet2,256,2,bk[2],trainable,'res_block2')
            print(resnet3)
            
            resnet4 = self.res_block(resnet3,512,2,bk[3],trainable,'res_block3')
            print(resnet4)
            
            avg_pool = tf.layers.average_pooling2d(resnet4,7,1,'VALID')
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
    
    def res_block(self,x,ch,stride,b_num,trainable,scope_name=''):
        with tf.variable_scope(scope_name):
            for i in range(b_num):
                if(i==0):
                    x = self.bottleneck(x,ch,stride,trainable)
                else:
                    x = self.bottleneck(x,ch,1,trainable)
            
            return x
    
    def bottleneck(self,x,ch,stride,trainable):
        pre_act = tf.layers.batch_normalization(x,training=trainable)
        pre_act = tf.nn.relu(pre_act)
        
        if(stride==2 or ch==x.shape[-1]):
            short_cut = tf.layers.conv2d(pre_act,ch*4,1,stride,padding='SAME')
            
        else:
            short_cut = x
        
        x2 = tf.layers.conv2d(x,ch,1,stride,padding='SAME',
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        x2 = tf.layers.batch_normalization(x2,training=trainable)
        x2 = tf.nn.relu(x2) 
        
        x2 = tf.layers.conv2d(x2,ch*4,3,1,padding='SAME',
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        x2 = tf.layers.batch_normalization(x2,training=trainable)
        x2 = tf.nn.relu(x2) 
        
        x2 = tf.layers.conv2d(x2,ch*4,1,1,padding='SAME',
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        x2 = tf.layers.batch_normalization(x2,training=trainable)
        x2 = tf.nn.relu(x2) 
        #print(short_cut,x2)
        return tf.add(x2,short_cut)
        

r = Resnet_v2(224,3,10)
        
        
        
        