import tensorflow as tf
import numpy as np

r = 16
bk_num = [4,6,3,2]
split_block = 8
sp_block_dim = 16


class Resnext:
    
    def __init__(self,img_size,channel,class_num=10,batch_size=16,trainable=True):
        self.k = 32
        self.drop_rate = 0.25
        self.grow_rate = 0.5
        self.batch_size = batch_size
        self.weight_init = tf.initializers.truncated_normal(0.0,0.01)
        self.weight_decay = tf.contrib.layers.l2_regularizer(0.0001)
        self.x = tf.placeholder(tf.float32,[self.batch_size,img_size,img_size,channel])
        self.class_num = class_num
        self.build_net(trainable)
        #self.code_87_predict(trainable)
        self.code_87_predict_2(trainable)
        #self.code_87_predict_3(trainable)
        
    def build_net(self,trainable):
        with tf.variable_scope('resnext',reuse=tf.AUTO_REUSE):
            
            resnext1 = tf.layers.conv2d(self.x,64,7,2,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext1 = tf.layers.batch_normalization(resnext1,training=trainable)
            self.resnext = tf.nn.relu(resnext1)
            self.resnext0 = tf.layers.max_pooling2d(self.resnext,3,2,padding='SAME')
            print(resnext1)
            self.resnext1 = self.res_block(self.resnext0,64,bk_num[0],False,trainable)
            #self.resnext1 = self.attention_layer2(self.resnext1,self.resnext1.shape[-1],'resnext1_atttention')
            print(self.resnext1)
            self.resnext2 = self.res_block(self.resnext1,128,bk_num[1],True,trainable)
            #self.resnext2 = self.attention_layer2(self.resnext2,self.resnext2.shape[-1],'resnext2_attention')
            print(self.resnext2)
            self.resnext3 = self.res_block(self.resnext2,256,bk_num[2],True,trainable)
            print(self.resnext3)
            
            self.resnext4 = self.res_block(self.resnext3,512,bk_num[3],True,trainable,False)
            print(self.resnext4)
            
            
    
    def batch_relu(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        return x
           
    def res_block(self,x,out_dim,block_num,flag,trainable,flag2=True):
        
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
            print(res_block.shape)
            res_block = self.cbam_module(res_block)
            
            
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
    
    def attention_layer2(self,x, ch,scope_name):
        with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
            print(x.shape)
            f = tf.layers.conv2d(x,ch//8,1,1)
            g = tf.layers.conv2d(x,ch//8,1,1)
            h = tf.layers.conv2d(x,ch,1,1)
            f = tf.reshape(f,[self.batch_size,-1,f.shape[-1]])
            g = tf.reshape(g,[self.batch_size,-1,g.shape[-1]])
            h = tf.reshape(h,[self.batch_size,-1,h.shape[-1]])

            s = tf.matmul(f,g,transpose_b=True)
            beta = tf.nn.softmax(s,axis=-1)

            o = tf.matmul(beta,h)
            gamma = tf.get_variable('gamma',[1],initializer=tf.constant_initializer(0.0))
            o = tf.reshape(o,shape=x.shape)
            x = gamma*o+x

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
    
    def cbam_module(self,inputs,reduction_ratio=0.5,name=""):
        with tf.variable_scope("cbam_"+name, reuse=tf.AUTO_REUSE):
            batch_size,hidden_num=inputs.get_shape().as_list()[0],inputs.get_shape().as_list()[3]
     
            maxpool_channel=tf.layers.max_pooling2d(inputs,(inputs.shape[1],inputs.shape[2]),1,padding='SAME')
            avgpool_channel=tf.layers.average_pooling2d(inputs,(inputs.shape[1],inputs.shape[2],1,padding='SAME'))
            
            maxpool_channel = tf.layers.Flatten(maxpool_channel)
            avgpool_channel = tf.layers.Flatten(avgpool_channel)
            
            mlp_1_max=tf.layers.dense(inputs=maxpool_channel,units=int(hidden_num*reduction_ratio),name="mlp_1",reuse=None,activation=tf.nn.relu)
            mlp_2_max=tf.layers.dense(inputs=mlp_1_max,units=hidden_num,name="mlp_2",reuse=None)
            mlp_2_max=tf.reshape(mlp_2_max,[batch_size,1,1,hidden_num])
     
            mlp_1_avg=tf.layers.dense(inputs=avgpool_channel,units=int(hidden_num*reduction_ratio),name="mlp_1",reuse=True,activation=tf.nn.relu)
            mlp_2_avg=tf.layers.dense(inputs=mlp_1_avg,units=hidden_num,name="mlp_2",reuse=True)
            mlp_2_avg=tf.reshape(mlp_2_avg,[batch_size,1,1,hidden_num])
     
            channel_attention=tf.nn.sigmoid(mlp_2_max+mlp_2_avg)
            channel_refined_feature=inputs*channel_attention
     
            maxpool_spatial=tf.reduce_max(inputs,axis=3,keepdims=True)
            avgpool_spatial=tf.reduce_mean(inputs,axis=3,keepdims=True)
            max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
            conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same", activation=None)
            spatial_attention=tf.nn.sigmoid(conv_layer)
     
            refined_feature=channel_refined_feature*spatial_attention
 
        return refined_feature

resnext = Resnext(256,3)