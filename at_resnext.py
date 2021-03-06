import tensorflow as tf
import numpy as np

r = 16
bk_num = [2,1,1,1]
split_block = 4
sp_block_dim = 64

class Resnext:
    
    def __init__(self,img_size,channel,class_num=10,trainable=True):
        self.weight_init = tf.initializers.truncated_normal(0.0,0.01)
        self.x = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
        self.class_num = class_num
        self.build_net(trainable)
        #self.code_87_predict(trainable)
        
    def build_net(self,trainable):
        with tf.variable_scope('resnext',reuse=tf.AUTO_REUSE):
            resnext1 = tf.layers.conv2d(self.x,64,7,2,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext1 = tf.layers.batch_normalization(resnext1,training=trainable)
            resnext1 = tf.nn.relu(resnext1)
            resnext1 = tf.layers.max_pooling2d(resnext1,3,2,padding='SAME')
            print(resnext1)
            self.resnext1 = self.res_block(resnext1,64,bk_num[0],False,trainable)
            print(self.resnext1)
            self.resnext2 = self.res_block(self.resnext1,128,bk_num[1],True,trainable)
            print(self.resnext2)
            self.resnext3 = self.res_block(self.resnext2,256,bk_num[2],True,trainable)
            print(self.resnext3)
            
            #self.resnext4 = self.res_block(self.resnext3,512,bk_num[3],True,trainable)
            #print(self.resnext4)
            
            avg_pool = tf.layers.average_pooling2d(self.resnext3,4,1,padding='SAME')
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
            
    def res_block(self,x,out_dim,block_num,flag,trainable):
        
        for i in range(block_num):
            if(flag==False):
                stride = 1
            else:
                stride = 2
                channel = x.shape[-1]//2
            res_block = self.merge_block(x,stride,trainable)
            res_block = self.transition_layer(res_block,out_dim,trainable)
            res_block = self.attention_layer(res_block)
            
            if(flag):
                pre_block = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                pre_block = tf.pad(pre_block,[[0,0],[0,0],[0,0],[channel,channel]])
                #pre_block = tf.layers.conv2d(x,out_dim,3,2,padding='SAME')
                flag = False
            else:
                pre_block = x
            #print(pre_block,res_block)
            #print(pre_block,res_block)
            x = pre_block+res_block
            x = tf.nn.relu(pre_block+res_block)
        return x
            
   
    def transition_layer(self,x,out_dim,trainable):
        x = tf.layers.conv2d(x,out_dim,1,1)
        x = tf.layers.batch_normalization(x,training=trainable)
        #print(out_dim)
        return x
    
    def attention_layer(self,x):
        avg_w = x.shape[1]
        avg_h = x.shape[2]
        #print(avg_w)
        ch = x.shape[3]
        at_x = tf.layers.average_pooling2d(x,(avg_w,avg_h),1,padding='VALID')
        at_x = tf.layers.flatten(at_x)
        at_x = tf.layers.dense(at_x,ch//r)
        at_x = tf.nn.relu(at_x)
        at_x = tf.layers.dense(at_x,ch)
        at_x = tf.sigmoid(at_x)
        at_x = tf.reshape(at_x,(-1,1,1,ch))
        x = x*at_x
        return x
        
    
    def merge_block(self,x,stride,trainable): 
        sp_block = list()
        for i in range(split_block):
            sp_block.append(self.bottleneck(x,stride,trainable))
        return self.Concatenation(sp_block)
            
    def bottleneck(self,x,stride,trainable):
        
        x = tf.layers.conv2d(x,sp_block_dim,1,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
       
        x = tf.layers.conv2d(x,sp_block_dim,3,stride,padding='SAME',
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        
        return x
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)