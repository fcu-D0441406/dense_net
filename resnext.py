import tensorflow as tf
import numpy as np

class Resnext:
    def __init__(self,img_size,channel,class_num,C,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.C = C
        self.build_net(C,trainable)
        
    
    def build_net(self,C,trainable):
        with tf.variable_scope('Resnext'):
            self.x = tf.placeholder(tf.float32,[None,self.img_size,self.img_size,self.channel])
            self. result = tf.placeholder(tf.float32,[None,10])
            resnext1 = tf.layers.conv2d(self.x,64,7,2,padding='SAME',activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext1 = tf.layers.batch_normalization(resnext1,trainable)
            resnext1 = tf.layers.max_pooling2d(resnext1,3,2,padding='SAME')
            self.resnext1 = self.resnext_block(resnext1,C,3,56,trainable)
            
            self.resnext2 = self.resnext_block(self.resnext1,C,4,28,trainable)
            self.resnext3 = self.resnext_block(self.resnext2,C,6,14,trainable)
            self.resnext4 = self.resnext_block(self.resnext3,C,3,7,trainable)
            print(self.resnext1)
            print(self.resnext2)
            print(self.resnext3)
            print(self.resnext4)
            avg_pool = tf.layers.average_pooling2d(self.resnext4,2,1)
            #print(avg_pool)
            flat = tf.contrib.layers.flatten(avg_pool)
            #print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            #print(self.prediction)
    
    def resnext_block(self,x,C,T,c,trainable):
        pre_block = x
        for j in range(T):
            for i in range(C):
                if(j==0):
                    block = self.resnext_bottleneck(pre_block,c,trainable)
                else:
                    block = tf.add(block,self.resnext_bottleneck(pre_block,c,trainable))
            if(pre_block.shape[3]!=block.shape[3]):
                pre_block = tf.layers.conv2d(pre_block,256,1,1,padding='SAME')
                pre_block = tf.add(block,pre_block)
            elif(pre_block.shape[1]!=block.shape[1]):
                pre_block = tf.layers.conv2d(pre_block,256,1,2,padding='SAME')
                #print(pre_block,block)
                pre_block = tf.add(block,pre_block)
            else:
                pre_block = tf.add(block,pre_block)
            pre_block = block
        return pre_block
    
    def resnext_bottleneck(self,x,c,trainable):
        if(c==x.shape[1]):
            stride = 1
        else:
            stride = 2
        net = tf.nn.relu(tf.layers.batch_normalization(x,training=trainable))
        net = tf.layers.conv2d(net,4,1,stride,padding='SAME')
        net = tf.nn.relu(tf.layers.batch_normalization(net,training=trainable))
        net = tf.layers.conv2d(net,4,3,1,padding='SAME')
        net = tf.nn.relu(tf.layers.batch_normalization(net,training=trainable))
        net = tf.layers.conv2d(net,256,1,1,padding='SAME')
        return net
            

if(__name__=='__main__'):
    r = Resnext(224,3,10,32)