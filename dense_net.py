import tensorflow as tf
import numpy as np
import os

class Dense_net:
    
    def __init__(self,img_size,channel,class_num,k,grow_rate,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[None,self.img_size,self.img_size,self.channel])
        
        self.build_net(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            dense1 = tf.layers.conv2d(self.x,self.k,7,2,padding='SAME')
            dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            print(dense1)
            dense2 = self.dense_block(dense1,6,trainable)
            print(dense2)
            
            dense3 = self.dense_block(dense2,12,trainable)
            print(dense3)
            '''
            dense4 = self.dense_block(dense3,24,trainable)
            print(dense4)
            dense5 = self.dense_block(dense4,16,trainable)
            print(dense5)
            '''
            
            avg_pool = tf.layers.average_pooling2d(dense3,2,1,padding='SAME')
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            print(flat)
            self.predict = tf.layers.dense(flat,self.class_num)
            print(self.predict)
    
    def dense_block(self,x,block_num,trainable):
        for i in range(block_num):
            if(i==0):
                x = self.bottleneck(x,trainable)
                #print(x)
            else:
                con_x = self.bottleneck(x,trainable)
                x = tf.concat([x,con_x],axis=3)
                #print(x)
        ch = (self.k*block_num)*self.grow_rate
        #print(ch)
        x = tf.layers.conv2d(x,ch,1,1,padding='SAME')
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        return x
    
    def bottleneck(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,1,1,padding='SAME')
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,3,1,padding='SAME')
        return x


if(__name__=='__main__'):
    ds = Dense_net(224,3,10,12,0.5)
    #print(resnet.predict)
