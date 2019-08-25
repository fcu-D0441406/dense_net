import tensorflow as tf
import numpy as np

class res2net:
    
    def __init__(self,img_size,channel,class_num=10,trainable=True):
        self.x = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
        self.class_num = class_num
        #self.build_net(trainable)
        self.build_pyramid_net(trainable)
        
    def build_normal_net(self,trainable):
        with tf.variable_scope('res2net',reuse=tf.AUTO_REUSE):
            net = tf.layers.conv2d(self.x,64,3,1,padding='SAME',
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            net = tf.layers.batch_normalization(net,training=trainable,momentum=0.9)
            net = tf.nn.relu(net)
            #net = tf.layers.conv2d(net,64,3,1,padding='SAME',
            #                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            #net = tf.layers.batch_normalization(net,training=trainable)
            #net = tf.nn.relu(net)
            
            #net = tf.layers.max_pooling2d(net,3,2,padding='SAME')
            print(net)
            
            self.net1 = self.res_block(net,64,3,trainable,False,scope='res2net_block0')
            print(self.net1)
            
            self.net2 = self.res_block(self.net1,128,4,trainable,scope='res2net_block1')
            print(self.net2)
            
            self.net3 = self.res_block(self.net2,256,6,trainable,scope='res2net_block2')
            print(self.net3)

            #self.net4 = self.res_block(self.net3,512,3,trainable)
            #print(self.net4)
            
            avg_pool = tf.layers.average_pooling2d(self.net3,8,1,padding='SAME')
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
            self.pre_softmax = tf.nn.softmax(self.prediction)
            print(self.pre_softmax)
    
    
    def res_block(self,x,out_dim,block_num,trainable,flag=True,scope=''):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            for i in range(block_num):                
                if(i==0 and flag==True):
                    x = self.bottleneck(x,out_dim,2,trainable,scope+str(i))
                else:
                    x = self.bottleneck(x,out_dim,1,trainable,scope+str(i))
            return x
    
    def bottleneck(self,x,ch,stride,trainable,scope=''):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            x = tf.layers.batch_normalization(x,training=trainable)
            #x = tf.nn.relu(x)
            if(stride==2):
                short_cut = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                short_cut = tf.pad(short_cut,[[0,0],[0,0],[0,0],[short_cut.shape[-1]//2,short_cut.shape[-1]//2]])
            else:
                short_cut = x
            
            x = tf.layers.conv2d(x,ch,3,stride,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            
            x = self.res2net_block(x,4,trainable)
            
            x = tf.layers.batch_normalization(x,training=trainable)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x,ch,1,1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            
            x = self.cbam_module(x)
            x = tf.layers.batch_normalization(x,training=trainable)
            return tf.add(x,short_cut)
    
    def build_pyramid_net(self,trainable):
        self.ch = 45
        self.add_channel = 30
        with tf.variable_scope('res2net',reuse=tf.AUTO_REUSE):
            net = tf.layers.conv2d(self.x,self.ch,3,1,padding='SAME',
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            net = tf.layers.batch_normalization(net,training=trainable,momentum=0.9)
            #net = tf.nn.relu(net)
            
            #net = tf.layers.conv2d(net,64,3,1,padding='SAME',
            #                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            #net = tf.layers.batch_normalization(net,training=trainable)
            #net = tf.nn.relu(net)
            
            #net = tf.layers.max_pooling2d(net,3,2,padding='SAME')
            print(net)
            
            self.net1 = self.res_pyramid_block(net,3,trainable,scope='res2net_block0')
            #self.net1 = self.transition_layer(self.net1,trainable,'transition1')
            print(self.net1)
            
            self.net2 = self.res_pyramid_block(self.net1,4,trainable,scope='res2net_block1')
            #self.net2 = self.transition_layer(self.net2,trainable,'transition2')
            print(self.net2)
            
            self.net3 = self.res_pyramid_block(self.net2,6,trainable,scope='res2net_block2')
            #self.net3 = self.transition_layer(self.net3,trainable,'transition3')
            print(self.net3)

            #self.net4 = self.res_block(self.net3,512,3,trainable)
            #print(self.net4)
            
            avg_pool = tf.layers.average_pooling2d(self.net3,4,1,padding='SAME')
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
            self.pre_softmax = tf.nn.softmax(self.prediction)
            print(self.pre_softmax)
            
    def res_pyramid_block(self,x,block_num,trainable,flag=True,scope=''):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            for i in range(block_num):                
                if(i==0 and flag==True):
                    x = self.pyramid_bottleneck(x,2,trainable,scope+str(i))
                else:
                    x = self.pyramid_bottleneck(x,1,trainable,scope+str(i))
            return x
    
    def pyramid_bottleneck(self,x,stride,trainable,scope=''):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            now_ch = x.shape[-1]
            self.ch+=self.add_channel
            
            x2 = tf.layers.batch_normalization(x,training=trainable)
            
            x2 = tf.layers.conv2d(x2,self.ch,3,stride,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            
            x2 = self.res2net_block(x2,3,trainable)
            
            x2 = tf.layers.batch_normalization(x2,training=trainable,momentum=0.9)
            x2 = tf.nn.relu(x2)
            x2 = tf.layers.conv2d(x2,self.ch,1,1,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            x2 = tf.layers.batch_normalization(x2,training=trainable,momentum=0.9)
            #x2 = self.SE_layer(x2,16)
            #x = self.cbam_module(x)
            
            add_channel = self.ch-now_ch
            short_cut = tf.pad(x,[[0,0],[0,0],[0,0],[0,add_channel]])
            if(stride==2):
                short_cut = tf.layers.average_pooling2d(short_cut,2,2,padding='SAME')
                
            #print(x2,short_cut)
            return tf.add(x2,short_cut)
    
    def transition_layer(self,x,trainable,scope_name=''):
        with tf.variable_scope(scope_name):
            x2 = tf.layers.batch_normalization(x,training=trainable)
            x2 = tf.nn.relu(x2)
            
            x2 = tf.layers.conv2d(x2,x2.shape[-1]//2,1,1,padding='SAME',
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            x2 = tf.layers.dropout(x2,0.25,training=trainable)
            x2 = tf.layers.average_pooling2d(x2,2,2,padding='SAME')
            return x2
    
    def res2net_block(self,x,s,trainable): 
        block = list()
        split_dim = int(x.shape[3])//s
        pre_block = None
        for i in range(s):
            now_block = x[:,:,:,i*split_dim:(i+1)*split_dim]
            if(i>1):
                now_block = tf.add(now_block,pre_block)     
            if(i>0):
                now_block = tf.layers.batch_normalization(now_block,training=trainable)
                now_block = tf.nn.relu(now_block)
                now_block = tf.layers.conv2d(now_block,split_dim,3,1,padding='SAME',
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            block.append(now_block)
            pre_block = now_block
        return self.Concatenation(block)
    
    def SE_layer(self,x,r):
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
        return x        
    
    def cbam_module(self,inputs,name="",reduction_ratio=16):
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
            mlp_2_max=tf.reshape(mlp_2_max,[-1,1,1,hidden_num])
     
            mlp_1_avg=tf.layers.dense(inputs=avgpool_channel,units=hidden_num//reduction_ratio,
                                      name="mlp_1",reuse=True,activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            mlp_2_avg=tf.layers.dense(inputs=mlp_1_avg,units=hidden_num,name="mlp_2",reuse=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            mlp_2_avg=tf.reshape(mlp_2_avg,[-1,1,1,hidden_num])
     
            channel_attention=tf.nn.sigmoid(mlp_2_max+mlp_2_avg)
            channel_refined_feature=tf.multiply(inputs,channel_attention)
            #print(channel_refined_feature)
            maxpool_spatial=tf.reduce_max(channel_refined_feature,axis=3,keepdims=True)
            avgpool_spatial=tf.reduce_mean(channel_refined_feature,axis=3,keepdims=True)
            max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
            conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(5, 5), padding="SAME", activation=None)
            spatial_attention=tf.nn.sigmoid(conv_layer)
     
            refined_feature=tf.multiply(channel_refined_feature,spatial_attention)
            #print(refined_feature)
            return refined_feature
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
'''
with tf.Graph().as_default(): 
    r = res2net(32,3)
'''