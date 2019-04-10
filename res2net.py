import tensorflow as tf
import numpy as np

r = 16
bk_num = [3,3,3]
split_block = 8
sp_block_dim = 16

class Resnext:
    
    def __init__(self,img_size,channel,class_num=10,trainable=True):
        self.x = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
        self.class_num = class_num
        self.build_net(trainable)
        
    def build_net(self,trainable):
        with tf.variable_scope('resnext',reuse=tf.AUTO_REUSE):
            resnext1 = tf.layers.conv2d(self.x,64,7,2,padding='SAME',
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext1 = tf.layers.batch_normalization(resnext1,training=trainable)
            resnext1 = tf.nn.relu(resnext1)
            #resnext1 = tf.layers.max_pooling2d(resnext1,3,2,padding='SAME')
            print(resnext1)
            
            self.resnext1 = self.res_block(resnext1,64,bk_num[0],trainable,True)
            print(self.resnext1)
            
            self.resnext2 = self.res_block(self.resnext1,128,bk_num[1],trainable,True)
            print(self.resnext2)
            
            self.resnext3 = self.res_block(self.resnext2,256,bk_num[2],trainable,True)
            print(self.resnext3)

            #self.resnext4 = self.res_block(self.resnext3,512,bk_num[3],trainable)
            #print(self.resnext4)
            
            avg_pool = tf.layers.average_pooling2d(self.resnext3,4,1,padding='SAME')
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
            
    def res_block(self,x,out_dim,block_num,trainable,flag=True):
        pre_block = x
        for i in range(block_num):                
            res_block = self.transition_layer(x,out_dim,trainable)
            res_block = self.scale_block(res_block,4,trainable)
            res_block = self.transition_layer(res_block,out_dim,trainable)
                    
            if(flag):
                res_block = self.attention_layer(res_block)
                pre_block = tf.layers.conv2d(pre_block,out_dim,3,2,padding='SAME')
                #pre_block = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                #pre_block = tf.pad(pre_block,[[0,0],[0,0],[0,0],[channel,channel]])
            else:
                pre_block = x
            x = pre_block+res_block
        return x
   
    def transition_layer(self,x,out_dim,trainable):
        x = tf.layers.batch_normalization(x,trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,out_dim,1,1,kernel_initializer=tf.contrib.layers.variance_scaling_initializer()) 
        return x
    
    def attention_layer(self,x):
        avg_w = x.shape[1]
        avg_h = x.shape[2]
        #print(avg_w)
        ch = x.shape[3]
        at_x = tf.layers.average_pooling2d(x,(avg_w,avg_h),1,padding='VALID')
        at_x = tf.layers.conv2d(at_x,ch//r,1,1,padding='SAME')
        at_x = tf.nn.relu(at_x)
        at_x = tf.layers.conv2d(at_x,ch,1,1,padding='SAME')
        at_x = tf.nn.sigmoid(at_x)
        x = tf.multiply(x,at_x)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        #print(x)
        return x
    
    def pre_act(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,x.shape[3],3,1,padding='SAME',kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        return x
    
    def scale_block(self,x,s,trainable): 
        block = list()
        split_dim = int(x.shape[3])//s
        pre_block = None
        for i in range(s):
            now_block = x[:,:,:,i*split_dim:(i+1)*split_dim]
            
            if(i>0):
                now_block = self.pre_act(pre_block,trainable)
            if(i>1):
                now_block = tf.add(now_block,pre_block)                 

            block.append(now_block)
            pre_block = now_block
        return self.Concatenation(block)
            
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)