import tensorflow as tf
import numpy as np
from oct_conv import *

class Res2net:
    
    def __init__(self,img_size,channel,batch_size=64,class_num=2,trainable=True):
        def normalize(x):

            r = tf.expand_dims((x[:,:,0] - tf.reduce_mean(x[:,:,0]))/tf.keras.backend.std(x[..., 0]),-1)
            g = tf.expand_dims((x[:,:,1] - tf.reduce_mean(x[:,:,1]))/tf.keras.backend.std(x[..., 1]),-1)
            b = tf.expand_dims((x[:,:,2] - tf.reduce_mean(x[:,:,2]))/tf.keras.backend.std(x[..., 2]),-1)
            return tf.concat([r,g,b],axis=-1)

        self.input = tf.placeholder(tf.uint8,[None,600,600,3],name='amd-clayton12-lubm-fv-camtek_inputLayer')
        inputs = tf.div(tf.cast(self.input,dtype=tf.float32),255.0)
        inputs = tf.cond(tf.equal(trainable,True),lambda:tf.map_fn(lambda x: tf.random_crop(x,size=(512,512,3)),inputs),
                        lambda:tf.image.crop_to_bounding_box(inputs,44,44,512,512))
        inputs = tf.map_fn(lambda x: normalize(x),inputs)
#         inputs = tf.cond(tf.equal(trainable,True),lambda:self.gaussian_noise_layer(inputs),lambda:inputs)
        
        self.inputs_magnifier = tf.image.crop_to_bounding_box(inputs,128,128,256,256)
        self.inputs = tf.image.resize_images(inputs,[256,256])

        print(self.input)
        self.expand = 4
        self.batch_size = batch_size
        self.class_num = class_num
        self.trainable = trainable
        self.build_net()
        self.cls_layer()

    def cls_layer(self):
        with tf.variable_scope('cls_layer'):
            predict_concat = tf.concat([self.prediction,
                                        self.prediction_GMP,
                                        self.prediction_DFL],axis=-1)
            
            self.two_cls_prediction = tf.layers.dense(predict_concat,self.class_num)
            self.two_cls_prediction = tf.nn.relu(self.two_cls_prediction)
            self.two_cls_prediction = tf.layers.batch_normalization(self.two_cls_prediction,training=self.trainable)
            
            
            self.two_cls_prediction = tf.layers.dense(self.two_cls_prediction,2)
            self.two_cls_pre_softmax = tf.nn.softmax(self.two_cls_prediction)    
            
    def svm_layer(self):
        with tf.variable_scope('svm_layer'):
            self.feature_concat = tf.concat([self.net4_decode_feature,
                                        self.net3_decode_feature],axis=-1)
            self.feature_concat = tf.layers.flatten(self.feature_concat)
                        
            self.A = tf.Variable(tf.random_normal(shape=[feature_concat.shape[-1],1]))
            self.b = tf.Variable(tf.random_normal(shape=[1,1],mean=0.5))
            self.svm_output = tf.subtract(tf.matmul(self.feature_concat, self.A), self.b)
        
    def gaussian_noise_layer(self,input_layer, std=0.1):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise
    
    def sobel(self, tensor):
        tensor = tf.nn.avg_pool(tensor, [1,5,5,1] , 1, 'SAME')
        tensor = tf.image.sobel_edges(tensor)
        tensor = tensor*tensor
        tensor = tf.math.reduce_sum(tensor, axis = -1)
        tensor = tf.sqrt(tensor)
        return tensor
    
    def blur_pooling(self,x,filt_size,stride=2,pool_type='avg',scope_name=''):
        with tf.variable_scope(scope_name):
            if(pool_type=='avg'):
                x = x
            elif(pool_type=='max'):
                x = tf.layers.max_pooling2d(x,2,1,padding='VALID')
            elif(pool_type=='conv'):
                x = tf.layers.conv2d(x,x.shape[-1],3,1,padding='SAME')

            if(filt_size==3):
                a = np.array([1., 2., 1.])
            elif(filt_size==5):
                a = np.array([1., 4., 6., 4., 1.])
            elif(filt_size==7):    
                a = np.array([1., 6., 15., 20., 15., 6., 1.])
            pad_off = 0
            pad_sizes = (filt_size-1)//2
            off = int((stride-1)/2.)
            channels = x.shape[-1]
            filt = a[:,None]*a[None,:]
            filt = filt/np.sum(filt)
            filt = filt[:,:,None,None]
            fi_ = np.tile(filt,[1,1,channels,1])
            w = tf.get_variable(name='anti_filter',dtype=tf.float32,shape=fi_.shape,trainable=False,initializer=tf.constant_initializer(fi_))
            w = tf.identity(w)
            x_ = x = tf.pad(x,[[0,0],[pad_sizes,pad_sizes],
                               [pad_sizes,pad_sizes],[0,0]])
            x_ = tf.nn.depthwise_conv2d(x_,w,(1,2,2,1),padding='VALID')
            return x_
    
    
    def build_net(self):
        with tf.variable_scope('res2net'):
            
#             self.inputs_magni = self.Concatenation([self.inputs, self.inputs_magnifier])
            
            self.inputs_magnifier_sobel = self.sobel(self.inputs_magnifier)
            self.inputs_sobel = self.sobel(self.inputs)
            self.inputs_ = self.Concatenation([self.inputs,self.inputs_magnifier,self.inputs_sobel,self.inputs_magnifier_sobel])
            net = tf.layers.conv2d(self.inputs_,64,3,1,padding='SAME',
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            net = self.batch_relu(net)
            
            net = tf.layers.conv2d(net,64,3,1,padding='SAME',
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            net = self.batch_relu(net)
            
            net = tf.layers.conv2d(net,64,3,1,padding='SAME',
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            net = self.blur_pooling(net,5,pool_type='conv',scope_name='conv_blur0')
            net = self.batch_relu(net)
            net = self.blur_pooling(net,5,pool_type='max',scope_name='max_blur0')
            print(net)
            
            self.net1_h,self.net1_l = self.first_oct_residual(net,64)
            for i in range(1,3):
                self.net1_h,self.net1_l = self.normal_oct_residual(self.net1_h,self.net1_l,64,scope_name='res_block0_'+str(i))
            print(self.net1_h,self.net1_l)


            self.net2_h,self.net2_l = self.stride_oct_residual(self.net1_h,self.net1_l,128,scope_name='stride_res_block1')
            for i in range(1,4):
                self.net2_h,self.net2_l = self.normal_oct_residual(self.net2_h,self.net2_l,128,scope_name='res_block1_'+str(i))
            print(self.net2_h,self.net2_l)


            self.net3_h,self.net3_l = self.stride_oct_residual(self.net2_h,self.net2_l,256,scope_name='stride_res_block2')
            for i in range(1,10):
                self.net3_h,self.net3_l = self.normal_oct_residual(self.net3_h,self.net3_l,256,scope_name='res_block2_'+str(i))
            self.net3 = self.last_oct_residual(self.net3_h,self.net3_l,256,'last_conv')
            for i in range(1,12):
                self.net3 = self.normal_residual(self.net3,256,scope_name='normal_res_block2_'+str(i))
            print('self.net3: ',self.net3)
            self.net4 = self.normal_stride_residual(self.net3,512,scope_name='stride_normal_block')
            for i in range(1,3):
                self.net4 = self.normal_residual(self.net4,512,scope_name='res_block3_'+str(i))

            print('self.net4: ',self.net4)

            avg_pool = tf.layers.average_pooling2d(self.net4,8,1,'VALID')
            flat = tf.layers.dropout(tf.layers.flatten(avg_pool),0.4,training=self.trainable)
            self.prediction = tf.layers.dense(flat,self.class_num,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            
            
            self.pre_softmax_ = tf.nn.softmax(self.prediction)
            print('prediction: ',self.prediction)

            ####DFL_layer
            k = 50
            height,width = self.net3.shape[1],self.net3.shape[2]
            DFL_conv = tf.layers.conv2d(self.net3,k*self.class_num,1,1,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            DFL_conv = self.batch_relu(DFL_conv)

            GMP_pool = tf.layers.max_pooling2d(DFL_conv,(height,width),1)

            GMP_pool_flat = tf.layers.dropout(tf.layers.flatten(GMP_pool),0.4,training=self.trainable)
            self.prediction_GMP = tf.layers.dense(GMP_pool_flat,self.class_num)
            self.pre_softmax_GMP = tf.nn.softmax(self.prediction_GMP)
            print(self.prediction_GMP)
            cross_channel_pooling = []
            print(GMP_pool_flat)
            GMP_split = tf.split(GMP_pool_flat,self.class_num,axis=1)
            for i in range(self.class_num):
                temp_cross = tf.reduce_mean(GMP_split[i],axis=-1)
                temp_cross = tf.reshape(temp_cross,[-1,1])
                cross_channel_pooling.append(temp_cross)
            self.prediction_DFL = tf.concat(cross_channel_pooling, axis=1) 
            self.pre_softmax_DFL = tf.nn.softmax(self.prediction_DFL)
            print(self.prediction_DFL)
            self.pre_softmax = 0.5*self.pre_softmax_+0.25*self.pre_softmax_GMP+0.25*self.pre_softmax_DFL
            
            
            
            
            self.net4_decode = tf.layers.conv2d(self.net4, 512, 1, 1, padding='SAME',
                                                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            self.net4_decode = tf.layers.batch_normalization(self.net4_decode,training=self.trainable)
            self.net4_decode = tf.nn.relu(self.net4_decode)
            self.net4_decode_feature = tf.nn.max_pool(self.net4_decode,8,1, padding='VALID')
            self.net4_decode = self.decode_layer(self.net4,256,scope_name='decode4_0') #16x16, 256
            self.net4_decode = self.decode_layer(self.net4_decode,128,scope_name='decode4_1') #32x32, 128
            self.net4_decode = self.decode_layer(self.net4_decode,64,scope_name='decode4_2') #64x64, 64
            self.net4_decode = self.decode_layer(self.net4_decode,16,scope_name='decode4_3') #128x128, 16
            self.net4_decode = self.decode_layer(self.net4_decode,3,scope_name='decode4_4') #256x256, 3
            print('net4_decode: ',self.net4_decode)
            
            self.net3_decode = tf.layers.conv2d(self.net3, 256, 1, 1, padding='SAME',
                                                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            self.net3_decode = tf.layers.batch_normalization(self.net3_decode,training=self.trainable)
            self.net3_decode = tf.nn.relu(self.net3_decode)
            self.net3_decode_feature = tf.nn.max_pool(self.net3_decode,16,1, padding='VALID')
            self.net3_decode = self.decode_layer(self.net3,128,scope_name='decode3_0') #32x32, 128
            self.net3_decode = self.decode_layer(self.net3_decode,64,scope_name='decode3_1') #64x64, 64
            self.net3_decode = self.decode_layer(self.net3_decode,16,scope_name='decode3_2') #128x128, 16
            self.net3_decode = self.decode_layer(self.net3_decode,3,scope_name='decode3_3') #256x256, 3
            print('net3_decode: ',self.net3_decode)
            
            def normalize2(x):

                r = tf.expand_dims((x[:,:,0] - tf.reduce_mean(x[:,:,0]))/tf.keras.backend.std(x[..., 0]),-1)
                g = tf.expand_dims((x[:,:,1] - tf.reduce_mean(x[:,:,1]))/tf.keras.backend.std(x[..., 1]),-1)
                b = tf.expand_dims((x[:,:,2] - tf.reduce_mean(x[:,:,2]))/tf.keras.backend.std(x[..., 2]),-1)
                return tf.concat([r,g,b],axis=-1)
            
            self.net3_decode = tf.nn.sigmoid(self.net3_decode)
            self.net4_decode = tf.nn.sigmoid(self.net4_decode)
            
            self.net3_decode = tf.map_fn(lambda x: normalize2(x),self.net3_decode)
            self.net4_decode = tf.map_fn(lambda x: normalize2(x),self.net4_decode)
            
            self.decode_output = 0.5*(self.net4_decode + self.net3_decode)
    
    def decode_layer(self,net,channel,scope_name=''):
        with tf.variable_scope(scope_name):
            net = tf.layers.conv2d_transpose(net, channel, 3, 2, padding='SAME',
                                                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            net = tf.layers.batch_normalization(net,training=self.trainable)
            net = tf.nn.relu(net)

            return net
    
    
    
    def first_oct_residual(self,x,ch,scope_name='first_oct_conv'):
        with tf.variable_scope(scope_name):
            
            h_data,l_data = first_oct_conv(x,ch,1,1,padding='SAME',scope_name='first')
            h_data,l_data = self.batch_relu(h_data),self.batch_relu(l_data)
            
            #h_data,l_data = self.oct_res2net_block(h_data,l_data,ch)
            h_data = self.res2net_block(h_data,h_data.shape[-1])
            l_data = self.res2net_block(l_data,l_data.shape[-1])
            
            h_data,l_data = normal_oct_conv(h_data,l_data,ch*self.expand,1,1,padding='SAME',scope_name='third')
            h_data,l_data = self.batch_relu(h_data,use_relu=False),self.batch_relu(l_data,use_relu=False)
            
            short_cut_h,short_cut_l = first_oct_conv(x,ch*self.expand,1,1,padding='SAME',scope_name='cut')
            short_cut_h,short_cut_l = self.batch_relu(short_cut_h,use_relu=False),self.batch_relu(short_cut_l,use_relu=False)
            
            #h_data,l_data = self.SE_module(h_data),self.SE_module(l_data)
            h_data,l_data = self.cbam_module(h_data,'h'),self.cbam_module(l_data,'l')
            return tf.nn.relu(h_data+short_cut_h),tf.nn.relu(l_data+short_cut_l)
    
    def normal_oct_residual(self,h_conv,l_conv,ch,scope_name=''):
        with tf.variable_scope(scope_name):
            short_cut_h,short_cut_l = h_conv,l_conv
            
            h_data,l_data = normal_oct_conv(h_conv,l_conv,ch,1,1,padding='SAME',scope_name='first')
            h_data,l_data = self.batch_relu(h_data),self.batch_relu(l_data)
            
#             h_data,l_data = self.oct_res2net_block(h_data,l_data,ch)
            h_data = self.res2net_block(h_data,h_data.shape[-1])
            l_data = self.res2net_block(l_data,l_data.shape[-1])
            
            h_data,l_data = normal_oct_conv(h_data,l_data,ch*self.expand,1,1,padding='SAME',scope_name='third')
            h_data,l_data = self.batch_relu(h_data,use_relu=False),self.batch_relu(l_data,use_relu=False)
            
            #h_data,l_data = self.SE_module(h_data),self.SE_module(l_data)
            h_data,l_data = self.cbam_module(h_data,'h'),self.cbam_module(l_data,'l')
            return tf.nn.relu(h_data+short_cut_h),tf.nn.relu(l_data+short_cut_l)
        
    def stride_oct_residual(self,h_conv,l_conv,ch,scope_name=''):
        with tf.variable_scope(scope_name):
            
            h_data,l_data = normal_oct_conv(h_conv,l_conv,ch,1,2,padding='SAME',scope_name='first')
            h_data,l_data = self.batch_relu(h_data),self.batch_relu(l_data)
            
#             h_data,l_data = self.oct_res2net_block(h_data,l_data,ch)
            h_data = self.res2net_block(h_data,h_data.shape[-1])
            l_data = self.res2net_block(l_data,l_data.shape[-1])
            
            h_data,l_data = normal_oct_conv(h_data,l_data,ch*self.expand,1,1,padding='SAME',scope_name='third')
            h_data,l_data = self.batch_relu(h_data,use_relu=False),self.batch_relu(l_data,use_relu=False)
            
            short_cut_h,short_cut_l = normal_oct_conv(h_conv,l_conv,ch*self.expand,1,2,padding='SAME',scope_name='cut')
            
            #h_data,l_data = self.SE_module(h_data),self.SE_module(l_data)
            h_data,l_data = self.cbam_module(h_data,'h'),self.cbam_module(l_data,'l')
            return tf.nn.relu(h_data+short_cut_h),tf.nn.relu(l_data+short_cut_l)
    
    def last_oct_residual(self,h_conv,l_conv,ch,scope_name='last_oct_conv'):
        with tf.variable_scope(scope_name):
            h_data,l_data = normal_oct_conv(h_conv,l_conv,ch,1,1,padding='SAME',scope_name='first')
            h_data,l_data = self.batch_relu(h_data),self.batch_relu(l_data)
            
#             h_data,l_data = self.oct_res2net_block(h_data,l_data,ch) 
            h_data = self.res2net_block(h_data,h_data.shape[-1])
            l_data = self.res2net_block(l_data,l_data.shape[-1])
            
            x2 = last_oct_conv(h_data,l_data,ch*self.expand,1,1,padding='SAME',scope_name='third')
            x2 = self.batch_relu(x2,use_relu=False)
            
            short_cut = last_oct_conv(h_conv,l_conv,ch*self.expand,1,1,padding='SAME',scope_name='short_cut')
            short_cut = self.batch_relu(short_cut,use_relu=False)
            
            #x2 = self.SE_module(x2)
            x2 = self.cbam_module(x2,'normal')
            return tf.nn.relu(x2+short_cut)
        
    def normal_residual(self,x,ch,scope_name=''):
        with tf.variable_scope(scope_name):
            short_cut = x
            
            x2 = tf.layers.conv2d(x,ch,1,1,padding='SAME',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            x2 = self.batch_relu(x2)
            
            x2 = self.res2net_block(x2,ch)
            
            x2 = tf.layers.conv2d(x2,ch*self.expand,1,1,padding='SAME',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            x2 = self.batch_relu(x2,use_relu=False)
            
            #x2 = self.SE_module(x2)
            x2 = self.cbam_module(x2,'normal')
            return tf.nn.relu(x2+short_cut)
    
    def normal_stride_residual(self,x,ch,scope_name=''):
        with tf.variable_scope(scope_name):
            short_cut = x
            
            x2 = blur_pooling(x,5,pool_type='conv',scope_name='conv_blur0')
            x = self.batch_relu(x)
            
            x2 = self.res2net_block(x2,ch)
            
            x2 = tf.layers.conv2d(x2,ch*self.expand,1,1,padding='SAME',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            x2 = self.batch_relu(x2,use_relu=False)
            
            short_cut = self.blur_pooling(short_cut,5,pool_type='avg',scope_name='avg_blur0')
            short_cut = tf.layers.conv2d(short_cut,ch*self.expand,1,1,padding='SAME',
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            short_cut = self.batch_relu(short_cut,use_relu=False)
            
            
            #x2 = self.SE_module(x2)
            x2 = self.cbam_module(x2,'normal')
            return tf.nn.relu(x2+short_cut)
    
    
    
    def oct_res2net_block(self,h_conv,l_conv,ch): 
        s = 4
        h_conv_block,l_conv_block = list(),list()
        split_dim = ch//4
        h_split_dim = int(h_conv.shape[-1])//s
        l_split_dim = int(l_conv.shape[-1])//s
        pre_h_conv_block,pre_l_conv_block = None,None
        
        for i in range(s):
            now_h_conv_block = h_conv[:,:,:,i*h_split_dim:(i+1)*h_split_dim]
            now_l_conv_block = l_conv[:,:,:,i*l_split_dim:(i+1)*l_split_dim]
            if(i>1):
                now_h_conv_block = tf.add(now_h_conv_block,pre_h_conv_block)
                now_l_conv_block = tf.add(now_l_conv_block,pre_l_conv_block)
                
            if(i>0):

                now_h_conv_block,now_l_conv_block = normal_oct_conv(now_h_conv_block,now_l_conv_block,split_dim,3,1,
                                                                             padding='SAME',scope_name='res2net_conv'+str(i))
                now_h_conv_block,now_l_conv_block = self.batch_relu(now_h_conv_block),self.batch_relu(now_l_conv_block)
                
            h_conv_block.append(now_h_conv_block)
            l_conv_block.append(now_l_conv_block)
            pre_h_conv_block,pre_l_conv_block = now_h_conv_block,now_l_conv_block
        return self.Concatenation(h_conv_block),self.Concatenation(l_conv_block)
    
    def res2net_block(self,x,ch): 
        s = 4
        block = list()
        split_dim = int(x.shape[-1])//s
        pre_block = None
        for i in range(s):
            now_block = x[:,:,:,i*split_dim:(i+1)*split_dim]
            if(i>1):
                now_block = tf.add(now_block,pre_block)
            if(i>0):
                now_block = tf.layers.conv2d(pre_block,split_dim,3,1,padding='SAME')
                now_block = self.batch_relu(now_block)
    
            block.append(now_block)
            pre_block = now_block
        return self.Concatenation(block)
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
    
    def batch_relu(self,x,use_relu=True,use_batch_norm=True):
        if(use_batch_norm):
            x = tf.layers.batch_normalization(x,training=self.trainable)
        if(use_relu):
            x = tf.nn.relu(x)
        return x
    
    def SE_module(self,x):
        r = 16
        
        avg_w = x.shape[1]
        avg_h = x.shape[2]
        #print(avg_w)
        ch = x.shape[3]
        at_x = tf.layers.average_pooling2d(x,(avg_w,avg_h),1,padding='VALID')
        at_x = tf.layers.conv2d(at_x,ch//r,1,1,padding='SAME',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        at_x = tf.nn.relu(at_x)
        at_x = tf.layers.conv2d(at_x,ch,1,1,padding='SAME',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
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
                                      name="mlp_1",reuse=True,activation=tf.nn.relu)
        
            mlp_2_avg=tf.layers.dense(inputs=mlp_1_avg,units=hidden_num,name="mlp_2",reuse=True)
            
            mlp_2_avg=tf.reshape(mlp_2_avg,[-1,1,1,hidden_num])
     
            channel_attention=tf.nn.sigmoid(mlp_2_max+mlp_2_avg)
            channel_refined_feature=tf.multiply(inputs,channel_attention)
            #print(channel_refined_feature)
            maxpool_spatial=tf.reduce_max(channel_refined_feature,axis=3,keepdims=True)
            avgpool_spatial=tf.reduce_mean(channel_refined_feature,axis=3,keepdims=True)
            max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
            conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="SAME",    
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            spatial_attention=tf.nn.sigmoid(conv_layer)
     
            refined_feature=tf.multiply(channel_refined_feature,spatial_attention)
            #print(refined_feature)
            return refined_feature
