import tensorflow as tf
import numpy as np
import os
class Dense_net:
    
    def __init__(self,img_size,channel,class_num,k,grow_rate,trainable=None,batch_size=64):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.drop_rate = 0.2
        self.batch_size = batch_size
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.img_size,self.img_size,self.channel])
        #self.x = self.x/255.0
        self.mrsa_init = tf.contrib.layers.variance_scaling_initializer()
        self.build_net(trainable)
        self.dense_cls_prediction()
        self.decoder(trainable)
        #self.upsample(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            dense1 = tf.layers.conv2d(self.x,2*self.k,3,1,padding='SAME',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            dense1 = tf.layers.batch_normalization(dense1,training=trainable)
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            
            self.dense1 = self.dense_block(dense1,4,trainable,scope='dense1')
            print(self.dense1)
            self.dense2= self.dense_block(self.dense1,6,trainable,scope='dense2')
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,8,trainable,trainsition=False,scope='dense3')
            print(self.dense3)
            '''
            self.dense4 = self.dense_block(self.dense3,16,False,trainable)
            print(self.dense4)
            '''
            avg_pool = tf.layers.average_pooling2d(self.dense3,4,1)
            print(avg_pool)
            flat = tf.contrib.layers.flatten(avg_pool)
            self.dense_flat = tf.layers.dense(flat,128)
            
    
    def dense_cls_prediction(self):
        self.prediction = tf.layers.dense(self.dense_flat,10)
        print(self.prediction)
    
    def decoder(self,trainable):
        decoder = tf.layers.dense(self.dense_flat,4*4*200)
        decoder = tf.nn.relu(decoder)
        decoder = tf.reshape(decoder,(self.batch_size,4,4,200))
        decoder = tf.layers.conv2d_transpose(decoder,72,3,2,padding='SAME')
        decoder = tf.layers.batch_normalization(decoder,training=trainable)
        decoder = tf.nn.relu(decoder)
        decoder = tf.layers.conv2d_transpose(decoder,48,3,2,padding='SAME')
        decoder = tf.layers.batch_normalization(decoder,training=trainable)
        decoder = tf.nn.relu(decoder)
        decoder = tf.layers.conv2d_transpose(decoder,3,3,2,padding='SAME')
        self.decoder_img = tf.nn.sigmoid(decoder)
    
    def dense_block(self,x,block_num,trainable,trainsition=True,scope=''):
        with tf.variable_scope(scope):
            for i in range(block_num):
                x = self.bottleneck(x,trainable)
                
            if(trainsition==True):
                x = self.trainsition_layer(x,trainable)
        return x
    
    def bottleneck(self,x,trainable):
        x2 = tf.layers.batch_normalization(x,training=trainable)
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2,self.k*4,1,1,padding='SAME',kernel_initializer=self.mrsa_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
        x2 = tf.layers.batch_normalization(x2,training=trainable)
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2,self.k,3,1,padding='SAME',kernel_initializer=self.mrsa_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
        #x2 = self.cbam_module(x2)
        x = self.Concatenation([x,x2])
        return x
    
    def trainsition_layer(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        C_channel = x.shape.as_list()[-1]
        C_channel = int(C_channel*self.grow_rate)
        x = tf.layers.conv2d(x,C_channel,1,1,padding='SAME',kernel_initializer=self.mrsa_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        x = tf.layers.dropout(x,self.drop_rate,training=trainable)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        x = self.cbam_module(x)
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
'''
if(__name__=='__main__'):
    ds = Dense_net(64,3,10,12,0.5)
    #print(resnet.predict)
'''
