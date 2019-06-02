import tensorflow as tf
import numpy as np


r = 16

class Dense_net:
    
    def __init__(self,img_size,channel,class_num,batch_size=64,k=16,grow_rate=1.0,trainable=None):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.batch_size = batch_size
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.img_size,self.img_size,self.channel])
        self.drop_rate = 0.2
        self.weight_init = tf.contrib.layers.variance_scaling_initializer()
        self.encoder_net(trainable)
        self.decoder_net(trainable)
 
        
    
    def encoder_net(self,trainable):
        with tf.variable_scope('dense_net'):
            #tf.contrib.layers.variance_scaling_initializer()
            dense1 = tf.layers.conv2d(self.x,2*self.k,7,1,padding='SAME',
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                                
            dense1 = tf.layers.batch_normalization(dense1,training=trainable)
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')#128
            print(dense1)
            self.dense1 = self.dense_block(dense1,4,trainable)
            self.dense1 = self.transition_layer(self.dense1,trainable,'dense1')#64
            print(self.dense1)
            
            self.dense2= self.dense_block(self.dense1,6,trainable)
            self.dense2 = self.transition_layer(self.dense2,trainable,'dense2')#32
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,8,trainable)
            self.dense3 = self.transition_layer(self.dense3,trainable,'dense3')#16
            print(self.dense3)
            self.ap_dense = self.ASAPP(self.dense3,trainable)
    
    def decoder_net(self,trainable):
        decoder_dense1 = self.upsample(self.ap_dense,4)
        concat_dense1 = tf.layers.conv2d(self.dense1,48,1,1,padding='SAME')
        decoder_dense1 = self.Concatenation([decoder_dense1,concat_dense1])
        decoder_dense1 = tf.layers.conv2d(decoder_dense1,256,3,1,padding='SAME')
        decoder_dense1 = tf.nn.relu(tf.layers.batch_normalization(decoder_dense1,training=trainable))
        decoder_dense1 = tf.layers.conv2d(decoder_dense1,256,3,1,padding='SAME')
        decoder_dense1 = tf.nn.relu(tf.layers.batch_normalization(decoder_dense1,training=trainable))
        
        decoder_dense2 = self.upsample(decoder_dense1,4)
        self.code_87_pre = tf.layers.conv2d(decoder_dense2,2,1,1,padding='SAME')
        self.code87_softmax = tf.nn.softmax(self.code_87_pre)
        print(self.code_87_pre)
        #decoder_dense1 = 
    
    def upsample(self,x,s):
        _, nh, nw, nx = x.get_shape().as_list()
        #x = tf.image.resize_nearest_neighbor(x, [nh * s, nw * s])
        x = tf.image.resize_bilinear(x, (nh * s, nw * s))
        return x
    
    def ASAPP(self,x,trainable):
        x1 = tf.layers.conv2d(x,128,1,1,padding='SAME')

        x3 = tf.layers.conv2d(x,128,3,1,padding='SAME')

        x5 = tf.layers.conv2d(x,128,5,1,padding='SAME')

        asapp_layer = self.Concatenation([x1,x3,x5])
        asapp_layer = tf.layers.batch_normalization(asapp_layer,training=trainable)
        asapp_layer = tf.layers.conv2d(asapp_layer,256,1,1,padding='SAME')
        return asapp_layer
        
    def dense_block(self,x,block_num,trainable):
        for i in range(block_num):
            x = self.bottleneck(x,trainable)
        return x
    
    def transition_layer(self,x,trainable,scope_name=''):
        with tf.variable_scope(scope_name):
            x2 = tf.layers.batch_normalization(x,training=trainable)
            x2 = tf.nn.relu(x2)
            x2 = tf.layers.conv2d(x2,x2.shape[-1],1,1,padding='SAME',
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
            x2 = tf.layers.max_pooling2d(x2,2,2,padding='SAME')
            return x2
    
    def bottleneck(self,x,trainable):
        x2 = tf.layers.batch_normalization(x,training=trainable)
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2,self.k*4,1,1,padding='SAME',
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
        
        x2 = tf.layers.batch_normalization(x2,training=trainable)
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2,self.k,3,1,padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
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
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)