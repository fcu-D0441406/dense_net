import tensorflow as tf
import numpy as np
import os
import cv2

r = 16

class Dense_net:
    
    def __init__(self,img_size,channel,class_num,batch_size=64,k=16,grow_rate=0.5,trainable=None):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.batch_size = batch_size
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.img_size,self.img_size,self.channel])
        self.drop_rate = 0.25
        self.weight_init = tf.contrib.layers.variance_scaling_initializer()
        self.build_net(trainable)
        self.dense_cls_prediction()
        #self.dense_local_prediction()
        self.decoder(trainable)
        #self.build_net_v2(trainable)
        
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            #tf.contrib.layers.variance_scaling_initializer()
            dense1 = tf.layers.conv2d(self.x,2*self.k,7,2,padding='SAME',
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                                
            dense1 = tf.layers.batch_normalization(dense1,training=trainable)
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            print(dense1)
            self.dense1 = self.dense_block(dense1,6,trainable)
            self.dense1 = self.transition_layer(self.dense1,trainable,'dense1')
            print(self.dense1)
            
            self.dense2= self.dense_block(self.dense1,12,trainable)
            self.dense2 = self.transition_layer(self.dense2,trainable,'dense2')
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,24,trainable)
            self.dense3 = self.transition_layer(self.dense3,trainable,'dense3')
            print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,16,trainable)
            #self.dense4 = self.transition_layer(self.dense4,trainable,'dense4')
            print(self.dense4)
            
            avg_pool = tf.layers.average_pooling2d(self.dense4,8,1,'VALID')
            flat = tf.layers.flatten(avg_pool)
            self.dense_flat = tf.layers.dense(flat,64)
            
    
    def dense_cls_prediction(self):
        self.prediction = tf.layers.dense(self.dense_flat,2)
        self.pre_softmax = tf.nn.softmax(self.prediction)
        print(self.prediction)
    
    def decoder(self,trainable):
        with tf.variable_scope('decoder'):
            decoder = tf.layers.dense(self.dense_flat,8*8*512)
            decoder = tf.layers.batch_normalization(decoder,training=trainable)
            decoder = tf.nn.relu(decoder)
            decoder = tf.reshape(decoder,(self.batch_size,8,8,512))
            decoder = tf.layers.conv2d_transpose(decoder,256,3,2,padding='SAME')
            decoder = tf.layers.batch_normalization(decoder,training=trainable)
            decoder = tf.nn.relu(decoder)
            decoder = tf.layers.conv2d_transpose(decoder,128,3,2,padding='SAME')
            decoder = tf.layers.batch_normalization(decoder,training=trainable)
            decoder = tf.nn.relu(decoder)
            decoder = tf.layers.conv2d_transpose(decoder,64,3,2,padding='SAME')
            decoder = tf.layers.batch_normalization(decoder,training=trainable)
            decoder = tf.nn.relu(decoder)
            decoder = tf.layers.conv2d_transpose(decoder,64,3,2,padding='SAME')
            decoder = tf.layers.batch_normalization(decoder,training=trainable)
            decoder = tf.nn.relu(decoder)
            decoder = tf.layers.conv2d_transpose(decoder,3,3,2,padding='SAME')
            self.decoder_img = tf.nn.sigmoid(decoder)
            print(self.decoder_img)
    
    
    def build_net_v2(self,trainable):
        with tf.variable_scope('dense_net'):
            #tf.contrib.layers.variance_scaling_initializer()
            dense1 = self.net_v2_f_block(self.x,trainable)

            print(dense1)
            self.dense1 = self.dense_block(dense1,6,trainable)
            self.dense1 = self.transition_layer_v2(self.dense1,trainable,'dense1')
            print(self.dense1)
            
            self.dense2= self.dense_block(self.dense1,12,trainable)
            self.dense2 = self.transition_layer_v2(self.dense2,trainable,'dense2')
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,24,trainable)
            self.dense3 = self.transition_layer_v2(self.dense3,trainable,'dense3')
            print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,16,trainable)
            #self.dense4 = self.transition_layer(self.dense4,trainable,'dense4')
            print(self.dense4)
            
            avg_pool = tf.layers.average_pooling2d(self.dense4,8,1,'VALID')
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
            self.pre_softmax = tf.nn.softmax(self.prediction)
            print(self.prediction)
    
    def net_v2_f_block(self,x,trainable):
        f_x = tf.layers.conv2d(x,16,7,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        f_x = self.batch_relu(f_x,trainable)
        
        f2_x = self.batch_relu(f_x,trainable)
        f2_x = tf.layers.conv2d(f2_x,16,3,2,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        
        f2_x = self.batch_relu(f2_x,trainable)
        f2_x = tf.layers.conv2d(f2_x,16,3,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        
        f_x = tf.layers.conv2d(f_x,16,3,2,padding='SAME',
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        f2_x = f_x+f2_x
        
        f3_x = self.batch_relu(f2_x,trainable)
        f3_x = tf.layers.conv2d(f3_x,32,3,2,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        
        f3_x = self.batch_relu(f3_x,trainable)
        f3_x = tf.layers.conv2d(f3_x,32,3,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        
        
        f2_x = tf.layers.conv2d(f2_x,32,3,2,padding='SAME',
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        f3_x = f2_x+f3_x
        
        return f3_x
    
    def dense_block(self,x,block_num,trainable):
        for i in range(block_num):
            x = self.bottleneck(x,trainable)
        return x
    
    def transition_layer(self,x,trainable,name=''):
        with tf.variable_scope(name):
            x = tf.layers.batch_normalization(x,training=trainable)
            x = tf.nn.relu(x)
            in_channel = int(x.shape[-1])*self.grow_rate
            x = tf.layers.conv2d(x,in_channel,1,1,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            #x = tf.layers.dropout(x,self.drop_rate,training=trainable)
            x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
            x = self.cbam_module(x)
        return x
    
    def transition_layer_v2(self,x,trainable,name=''):
        with tf.variable_scope(name):
            x = tf.layers.batch_normalization(x,training=trainable)
            x = tf.nn.relu(x)
            in_channel = int(x.shape[-1])*self.grow_rate
            x1 = tf.layers.conv2d(x,in_channel,3,2,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),padding='SAME')
            x2 = tf.layers.conv2d(x,in_channel,5,2,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),padding='SAME')
            x = self.Concatenation([x1,x2])
            x = tf.layers.conv2d(x,in_channel,1,1,
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),padding='SAME')
            x = self.batch_relu(x,trainable)
            x = self.cbam_module(x)
        return x
    
    def bottleneck(self,x,trainable):
        x2 = tf.layers.batch_normalization(x,training=trainable)
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2,self.k*4,1,1,padding='SAME',
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        #x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
        
        x2 = tf.layers.batch_normalization(x2,training=trainable)
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2,self.k,3,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        #x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
        x = self.Concatenation([x,x2])
        return x
    
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
            conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="SAME", activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            spatial_attention=tf.nn.sigmoid(conv_layer)
     
            refined_feature=tf.multiply(channel_refined_feature,spatial_attention)
            #print(refined_feature)
            return refined_feature
    
    def batch_relu(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        return x
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
    