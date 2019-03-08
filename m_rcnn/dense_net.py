import tensorflow as tf
import numpy as np
import os
import cv2



class Dense_net:
    
    def __init__(self,img_size,channel,class_num,k,grow_rate,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[None,self.img_size,self.img_size,self.channel])
        self.drop_rate = 0.2
        self.weight_init1 = tf.initializers.truncated_normal(0.0,0.01)
        self.weight_init = tf.initializers.truncated_normal(0.0,0.01)
        self.weight_decay = tf.contrib.layers.l2_regularizer(0.0001)
        self.a_num = 1
        self.ratio_num = 3
        self.anchor_num = self.a_num*self.ratio_num
        self.build_net(trainable)
        
        self.concat_predict()
        #self.best_fg_score,self.best_index = self.get_best_fg()
        #print(self.best_fg_score,self.best_index)
        
        #self.upsample(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            
            #tf.contrib.layers.variance_scaling_initializer()
            dense1 = tf.layers.conv2d(self.x,2*self.k,7,2,padding='SAME',activation=tf.nn.relu)
            dense1 = tf.contrib.layers.batch_norm(dense1,is_training=trainable)
            #dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            
            self.dense1 = self.dense_block(dense1,6,True,trainable)
            self.dense1 = self.transition_layer(self.dense1,trainable)
            print(self.dense1)
            self.dense2= self.dense_block(self.dense1,6,False,trainable)
            self.dense2 = self.transition_layer(self.dense2,trainable)
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,12,False,trainable)
            self.dense3 = self.transition_layer(self.dense3,trainable)
            print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,16,False,trainable)
            self.dense4 = self.transition_layer(self.dense4,trainable)
            print(self.dense4)

            self.pre0 = self.fpn_net(self.dense4,None,True)
            self.pre1 = self.fpn_net(self.pre0,self.dense3,False)
            self.pre2 = self.fpn_net(self.pre1,self.dense2,False)
            self.pre3 = self.fpn_net(self.pre2,self.dense1,False)
            ac = tf.nn.relu
            self.pre0 = tf.layers.conv2d(self.pre0,256,3,1,padding='SAME',kernel_initializer=self.weight_init,
                                          activation=ac,kernel_regularizer=self.weight_decay) 
            self.pre0 = tf.layers.batch_normalization(self.pre0)
            self.pre1 = tf.layers.conv2d(self.pre1,256,3,1,padding='SAME',kernel_initializer=self.weight_init,
                                          activation=ac,kernel_regularizer=self.weight_decay) 
            self.pre1 = tf.layers.batch_normalization(self.pre1)
            self.pre2 = tf.layers.conv2d(self.pre2,256,3,1,padding='SAME',kernel_initializer=self.weight_init,
                                          activation=ac,kernel_regularizer=self.weight_decay) 
            self.pre2 = tf.layers.batch_normalization(self.pre2)
            self.pre3 = tf.layers.conv2d(self.pre3,256,3,1,padding='SAME',kernel_initializer=self.weight_init,
                                          activation=ac,kernel_regularizer=self.weight_decay) 
            self.pre3 = tf.layers.batch_normalization(self.pre3)
            
            self.rpn0,self.fg0,self.fg0_score,self.box0 = self.rpn_net(self.pre0)
            self.rpn1,self.fg1,self.fg1_score,self.box1 = self.rpn_net(self.pre1)
            self.rpn2,self.fg2,self.fg2_score,self.box2 = self.rpn_net(self.pre2)
            self.rpn3,self.fg3,self.fg3_score,self.box3 = self.rpn_net(self.pre3)
            
            print(self.fg0,self.fg0_score,self.box0)
            print(self.fg1,self.fg1_score,self.box1)
            print(self.fg2,self.fg2_score,self.box2)
            print(self.fg3,self.fg3_score,self.box3)
            
            
    def dense_block(self,x,block_num,first,trainable):
        layer_concat = list()
        layer_concat.append(x)
        x = self.bottleneck(x,trainable)
        layer_concat.append(x)
        for i in range(block_num-1):
            x = self.Concatenation(layer_concat)
            x = self.bottleneck(x,trainable)
            layer_concat.append(x)
        x = self.Concatenation(layer_concat)
        return x
    
    def transition_layer(self,x,trainable):
        x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        x = tf.nn.relu(x)
        in_channel = int(x.shape[-1])*0.5
        print(type(in_channel))
        x = tf.layers.conv2d(x,in_channel,1,1)
        x = tf.layers.dropout(x,self.drop_rate,training=trainable)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        return x
    
    def bottleneck(self,x,trainable):
        x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k*4,1,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.dropout(x,self.drop_rate)
        x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,3,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.dropout(x,self.drop_rate)
        return x
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
    
    def fpn_net(self,pre_net,now_net,first):
        if(first==True):
            net = tf.layers.conv2d(pre_net,256,3,1,padding='SAME',activation=tf.nn.relu,
                                  kernel_initializer=self.weight_init,
                                  kernel_regularizer=self.weight_decay)
        else:
            pre = tf.layers.conv2d_transpose(pre_net,256,3,2,padding='SAME',
                                                    kernel_initializer=self.weight_init,
                                                    kernel_regularizer=self.weight_decay)
            now = tf.layers.conv2d(now_net,256,1,1,padding='SAME',activation=tf.nn.relu,
                                          kernel_initializer=self.weight_init,
                                          kernel_regularizer=self.weight_decay)
            net = tf.add(pre,now)
            
        return net
    
    def rpn_net(self,net):
        init0 = tf.initializers.random_normal(0,0.01)
        init1 = tf.initializers.random_uniform(0.0,1.0)
        init2 = tf.initializers.zeros()
        #kernel_initializer=init0
        rpn = tf.layers.conv2d(net,512,3,1,padding='SAME',kernel_initializer=self.weight_init1,activation=tf.nn.relu,
                                          kernel_regularizer=self.weight_decay) 
        fg_pre = tf.layers.conv2d(rpn,self.anchor_num*2,1,1,kernel_initializer=self.weight_init,
                                          kernel_regularizer=self.weight_decay) 
        box_pre = tf.layers.conv2d(rpn,self.anchor_num*4,1,1,kernel_initializer=self.weight_init,
                                          kernel_regularizer=self.weight_decay) 

        #print(fg_pre)
        fg_pre = tf.reshape(fg_pre,(-1,fg_pre.shape[1],fg_pre.shape[2],self.anchor_num,2))
        box_pre = tf.reshape(box_pre,(-1,box_pre.shape[1],box_pre.shape[2],self.anchor_num,4))
        fg_pre_score = tf.nn.softmax(fg_pre)
        #print(fg_pre)
        fg_pre = tf.reshape(fg_pre,(-1,fg_pre.shape[1]*fg_pre.shape[2]*self.anchor_num,2))
        box_pre = tf.reshape(box_pre,(-1,box_pre.shape[1]*box_pre.shape[2]*self.anchor_num,4))
        fg_pre_score = tf.reshape(fg_pre_score,(-1,fg_pre_score.shape[1]*fg_pre_score.shape[2]*self.anchor_num,2))
        #print(fg_pre)
        return rpn,fg_pre,fg_pre_score,box_pre
    
    def concat_predict(self):
        self.all_fg = tf.concat([self.fg0,self.fg1,self.fg2,self.fg3],axis=1)
        self.all_fg_score = tf.concat([self.fg0_score,self.fg1_score,
                                       self.fg2_score,self.fg3_score],axis=1)
        self.all_box = tf.concat([self.box0,self.box1,self.box2,self.box3],axis=1)
        
        print(self.all_fg,self.all_fg_score,self.all_box)
        
        
    def get_best_fg(self):
        k = self.all_fg.shape[1]
        best_value = tf.nn.top_k(self.all_fg_score,k).values
        best_index = tf.nn.top_k(self.all_fg_score,k).indices
        return best_value,best_index
    
    
    def upsample(self,trainable):
        self.deconv1 = self.up_dense_block(self.dense3,self.dense4,24,trainable)
        print(self.deconv1)
        self.deconv2 = self.up_dense_block(self.dense2,self.deconv1,12,trainable)
        print(self.deconv2)
        self.deconv3 = self.up_dense_block(self.dense1,self.deconv2,6,trainable)
        print(self.deconv3)
        self.deconv4 = tf.layers.batch_normalization(self.deconv3,training=trainable)
        self.deconv4 = tf.nn.relu(self.deconv4)
        self.deconv4 = tf.layers.conv2d_transpose(self.deconv4,self.class_num,5,4,padding='SAME')
        print(self.deconv4)
    
    def up_dense_block(self,pre_net,x,block_num,trainable):
        for i in range(block_num):
            if(i==0):
                x = self.bottleneck(x,trainable)
                #print(x)
            else:
                con_x = self.bottleneck(x,trainable)
                x = tf.concat([x,con_x],axis=3)
                #print(x)
        ch = int((self.k*block_num)*self.grow_rate)
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        
        x = tf.layers.conv2d_transpose(x,ch,3,2,padding='SAME')
        #print(pre_net)
        #print(x)
        x = tf.concat([x,pre_net],axis=3)
        return x