import tensorflow as tf
import numpy as np

r = 16
bk_num = [3,4,6,3]
split_block = 8
sp_block_dim = 16


class Resnext:
    
    def __init__(self,img_size,channel,class_num=10,batch_size=64,trainable=True):
        
        self.batch_size = batch_size
        self.weight_init = tf.initializers.truncated_normal(0.0,0.01)
        self.weight_decay = tf.contrib.layers.l2_regularizer(0.0001)
        self.x = tf.placeholder(tf.float32,[self.batch_size,img_size,img_size,channel])
        self.class_num = class_num
        #self.build_net(trainable)
        self.build_net_v2(trainable)
        
    def build_net(self,trainable):
        with tf.variable_scope('resnext',reuse=tf.AUTO_REUSE):
            
            resnext = tf.layers.conv2d(self.x,64,7,2,padding='SAME',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext = tf.layers.batch_normalization(resnext,training=trainable)
            resnext = tf.nn.relu(resnext)
            resnext = tf.layers.max_pooling2d(resnext,3,2,padding='SAME')
            print(resnext)
            self.resnext1 = self.res_block(resnext,64,bk_num[0],False,trainable,'block1')
            #self.resnext1 = self.attention_layer2(self.resnext1,self.resnext1.shape[-1],'resnext1_atttention')
            print(self.resnext1)
            self.resnext2 = self.res_block(self.resnext1,128,bk_num[1],True,trainable,'block2')
            #self.resnext2 = self.attention_layer2(self.resnext2,self.resnext2.shape[-1],'resnext2_attention')
            print(self.resnext2)
            self.resnext3 = self.res_block(self.resnext2,256,bk_num[2],True,trainable,'block3')
            print(self.resnext3)
            
            #self.resnext4 = self.res_block(self.resnext3,512,bk_num[3],True,trainable,'block4',False)
            #print(self.resnext4)
            
            avg_pool = tf.layers.average_pooling2d(self.resnext3,4,1)
            #print(avg_pool)
            flat = tf.contrib.layers.flatten(avg_pool)
            #print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
    
    def build_net_v2(self,trainable):
        with tf.variable_scope('resnext',reuse=tf.AUTO_REUSE):
            
            resnext = self.net_v2_f_block(self.x,trainable)
            print(resnext)
            self.resnext1 = self.res_block(resnext,64,bk_num[0],True,trainable,'block1')
            #self.resnext1 = self.attention_layer2(self.resnext1,self.resnext1.shape[-1],'resnext1_atttention')
            print(self.resnext1)
            self.resnext2 = self.res_block(self.resnext1,128,bk_num[1],True,trainable,'block2')
            #self.resnext2 = self.attention_layer2(self.resnext2,self.resnext2.shape[-1],'resnext2_attention')
            print(self.resnext2)
            self.resnext3 = self.res_block(self.resnext2,256,bk_num[2],True,trainable,'block3')
            print(self.resnext3)
            
            #self.resnext4 = self.res_block(self.resnext3,512,bk_num[3],True,trainable,'block4',False)
            #print(self.resnext4)
            
            avg_pool = tf.layers.average_pooling2d(self.resnext3,4,1)
            #print(avg_pool)
            flat = tf.contrib.layers.flatten(avg_pool)
            #print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
    
    def net_v2_f_block(self,x,trainable):
        f_x = tf.layers.conv2d(x,16,7,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        f_x = self.batch_relu(f_x,trainable)
        
        f_x = self.res_block(f_x,16,2,False,trainable,'pre_block1')
        
        f_x = self.res_block(f_x,32,2,True,trainable,'pre_block2')
        
        return f_x
    
    def batch_relu(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        return x
           
    def res_block(self,x,out_dim,block_num,flag,trainable,name='',flag2=True):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            for i in range(block_num):
                if(flag==False):
                    stride = 1
                else:
                    if(flag2==True):
                        stride = 2
                    else:
                        stride = 1
                    channel = x.shape[-1]//2
                    
                res_block = self.merge_block(x,stride,trainable)
                #res_block = self.attention_layer(res_block)
                res_block = self.transition_layer(res_block,out_dim,trainable)
                res_block = self.cbam_module(res_block,str(i))
                #print(res_block)
                
                if(flag):
                    if(flag2==True):
                        pre_block = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                    else:
                        pre_block = x
                    pre_block = tf.pad(pre_block,[[0,0],[0,0],[0,0],[channel,channel]])
                    flag = False
                else:
                    pre_block = x
    
                x = tf.nn.relu(pre_block+res_block)
    
            return x
            
   
    def transition_layer(self,x,out_dim,trainable):
        x = tf.layers.conv2d(x,out_dim,1,1)
        x = tf.layers.batch_normalization(x,training=trainable)
        #print(out_dim)
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
    
    def cbam_module(self,inputs,name="",reduction_ratio=2):
        with tf.variable_scope("cbam_"+name, reuse=tf.AUTO_REUSE):
            batch_size,hidden_num=inputs.get_shape().as_list()[0],inputs.get_shape().as_list()[3]
     
            maxpool_channel=tf.layers.max_pooling2d(inputs,(inputs.shape[1],inputs.shape[2]),1,padding='VALID')
            avgpool_channel=tf.layers.average_pooling2d(inputs,(inputs.shape[1],inputs.shape[2]),1,padding='VALID')
            
            maxpool_channel = tf.layers.flatten(maxpool_channel)
            avgpool_channel = tf.layers.flatten(avgpool_channel)
            
            mlp_1_max=tf.layers.dense(inputs=maxpool_channel,units=hidden_num//reduction_ratio,
                                      name="mlp_1",reuse=None,activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            mlp_2_max=tf.layers.dense(inputs=mlp_1_max,units=hidden_num,name="mlp_2",reuse=None,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            mlp_2_max=tf.reshape(mlp_2_max,[batch_size,1,1,hidden_num])
     
            mlp_1_avg=tf.layers.dense(inputs=avgpool_channel,units=hidden_num//reduction_ratio,
                                      name="mlp_1",reuse=True,activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            mlp_2_avg=tf.layers.dense(inputs=mlp_1_avg,units=hidden_num,name="mlp_2",reuse=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            mlp_2_avg=tf.reshape(mlp_2_avg,[batch_size,1,1,hidden_num])
     
            channel_attention=tf.nn.sigmoid(mlp_2_max+mlp_2_avg)
            channel_refined_feature=tf.multiply(inputs,channel_attention)
            #print(channel_refined_feature)
            maxpool_spatial=tf.reduce_max(channel_refined_feature,axis=3,keepdims=True)
            avgpool_spatial=tf.reduce_mean(channel_refined_feature,axis=3,keepdims=True)
            max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
            conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="SAME", activation=None)
            spatial_attention=tf.nn.sigmoid(conv_layer)
     
            refined_feature=tf.multiply(channel_refined_feature,spatial_attention)
            #print(refined_feature)
            return refined_feature

#resnext = Resnext(256,3)