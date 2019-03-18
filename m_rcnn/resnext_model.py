import tensorflow as tf
import numpy as np

block_num = 1
split_block = 8
sp_block_dim = 16

class Resnext:
    
    def __init__(self,img_size,channel,class_num=10,trainable=True):
        self.x = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
        self.class_num = class_num
        self.build_net(trainable)
        
    def build_net(self,trainable):
        with tf.variable_scope('resnext'):
            resnext1 = tf.layers.conv2d(self.x,64,7,2,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext1 = tf.layers.batch_normalization(resnext1,training=trainable)
            resnext1 = tf.nn.relu(resnext1)
            resnext1 = tf.layers.max_pooling2d(resnext1,3,2,padding='SAME')
            print(resnext1)
            self.resnext1 = self.res_block(resnext1,64,trainable)
            print(self.resnext1)
            self.resnext2 = self.res_block(self.resnext1,128,trainable)
            print(self.resnext2)
            self.resnext3 = self.res_block(self.resnext2,256,trainable)
            print(self.resnext3)
            
            self.resnext4 = self.res_block(self.resnext3,512,trainable)
            print(self.resnext4)
            '''
            avg_pool = tf.layers.average_pooling2d(self.resnext4,4,1,padding='SAME')
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
            '''
            
    def res_block(self,x,out_dim,trainable):
        
        for i in range(block_num):
            flag = False
            if(x.shape[3]==out_dim):
                flag = False
                stride = 1
            else:
                flag = True
                stride = 2
                channel = x.shape[-1]//2
            res_block = self.merge_block(x,stride,trainable)
            res_block = self.transition_layer(res_block,out_dim,trainable)
            
            if(flag):
                '''
                tf.pad->用0填補到跟x的channel相等
                '''
                pre_block = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                pre_block = tf.pad(pre_block,[[0,0],[0,0],[0,0],[channel,channel]])
            else:
                pre_block = x
            x = tf.nn.relu(pre_block+res_block)
        return x
            
   
    def transition_layer(self,x,out_dim,trainable):
        x = tf.layers.conv2d(x,out_dim,1,1)
        x = tf.layers.batch_normalization(x,trainable)

        return x
    
    def merge_block(self,x,stride,trainable): 
        sp_block = list()
        for i in range(split_block):
            sp_block.append(self.bottleneck(x,stride,trainable))
        return self.Concatenation(sp_block)
            
    def bottleneck(self,x,stride,trainable):
        #x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        x = tf.layers.conv2d(x,sp_block_dim,1,stride,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
       

        #x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        x = tf.layers.conv2d(x,sp_block_dim,3,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        
        return x
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
    
#r = Resnext(28,3,10)