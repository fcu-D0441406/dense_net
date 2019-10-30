import tensorflow as tf
import numpy as np

class Res2net:
    
    def __init__(self,batch_size=1,class_num=2,trainable=True):
        
        self.img_size = 256
        self.batch_size = batch_size
        self.class_num = class_num
        self.trainable = trainable
        
        self.C6_feature_center = tf.placeholder(tf.float32,[batch_size,16,16,2])
        self.P5_feature_center = tf.placeholder(tf.float32,[batch_size,32,32,2])
        self.P4_feature_center = tf.placeholder(tf.float32,[batch_size,64,64,2])
        
        self.input = tf.placeholder(tf.uint8,[batch_size,256,256,3],name='amd-clayton12-lubm-fv-camtek_inputLayer')
        self.ground_truth = tf.placeholder(tf.float32,[batch_size,None,5])
        self.ground_truths = self.ground_truth
        self.inputs = tf.div(tf.cast(self.input,dtype=tf.float32),255.0)

        print(self.input)
        
        self.build_net()
    
    def blur_pooling(self,x,filt_size,stride=2,pool_type='avg',scope_name=''):
        with tf.variable_scope(scope_name):
            if(pool_type=='avg'):
                x = x
            elif(pool_type=='max'):
                x = tf.layers.max_pooling2d(x,2,1,padding='VALID')
            elif(pool_type=='conv'):
                x = tf.layers.conv2d(x,x.shape[-1],3,1,padding='SAME')
                x = tf.layers.batch_normalization(x,training=self.trainable)
                x = tf.nn.relu(x)

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

    def cls_layer(self):
        with tf.variable_scope('two_cls'):
            predict_concat = tf.concat([self.prediction,
                                        self.prediction_GMP,
                                        self.prediction_DFL],axis=-1)
            
            self.two_cls_prediction = tf.layers.dense(predict_concat,self.class_num)
            self.two_cls_prediction = tf.nn.relu(predict_concat)
            self.two_cls_prediction = tf.layers.batch_normalization(self.two_cls_prediction,training=self.trainable)
            
            
            self.two_cls_prediction = tf.layers.dense(self.two_cls_prediction,2)
            self.two_cls_pre_softmax = tf.nn.softmax(self.two_cls_prediction)    
        
    def gaussian_noise_layer(self,input_layer, std=0.1):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise
    
    def build_net(self):
        
        with tf.variable_scope('res2net'):
            net = tf.layers.conv2d(self.inputs,64,3,1,padding='SAME',
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

            self.C3 = self.blur_pooling(net,5,pool_type='max',scope_name='max_blur0')

            self.C4 = self.resnet_block(self.C3,64,3,5,True,scope_name='res2net0')#64

            self.C5 = self.resnet_block(self.C4,128,4,5,scope_name='res2net1')#32

            self.C6 = self.resnet_block(self.C5,256,6,5,scope_name='res2net2')#16

            
            self.P5 = self.fpn_net(self.C5,self.C6)#32
            self.P4 = self.fpn_net(self.C4,self.P5)#64
            
            self.P5 = tf.layers.conv2d(self.P5,self.P5.shape[-1],1,1,padding='SAME')
            self.P5 = self.batch_relu(self.P5)
            
            self.P4 = tf.layers.conv2d(self.P4,self.P4.shape[-1],1,1,padding='SAME')
            self.P4 = self.batch_relu(self.P4)
            
            self.C6_reg_pre = self.build_reg_net(self.C6)
            self.C6_cs_pre,self.C6_cls_pre = self.build_cs_net(self.C6)
            #print(self.C6_reg_pre,self.C6_cs_pre,self.C6_cls_pre)
            
            self.P5_reg_pre = self.build_reg_net(self.P5)
            self.P5_cs_pre,self.P5_cls_pre = self.build_cs_net(self.P5)
            #print(self.P5_reg_pre,self.P5_cs_pre,self.P5_cls_pre)
            
            self.P4_reg_pre = self.build_reg_net(self.P4)
            self.P4_cs_pre,self.P4_cls_pre = self.build_cs_net(self.P4)
            #print(self.P4_reg_pre,self.P4_cs_pre,self.P4_cls_pre)
            
            self.C6_loss,self.P5_loss,self.P4_loss = 0,0,0
#             print(self.ground_truths[0,:,:,:4],self.C6_feature_center[0],self.C6_reg_pre[0])
#             print(self.ground_truths)
#             print(self.C6_feature_center)
#             print(self.C6_reg_pre)
            
            for i in range(self.batch_size):
                self.C6_reg_loss = self.C6_loss+self.reg_loss(self.ground_truths[i,:,:4],self.C6_feature_center[i],self.C6_reg_pre[i])
                self.P5_reg_loss = self.P5_loss+self.reg_loss(self.ground_truths[i,:,:4],self.P5_feature_center[i],self.P5_reg_pre[i])
                self.P4_reg_loss = self.P4_loss+self.reg_loss(self.ground_truths[i,:,:4],self.P4_feature_center[i],self.P4_reg_pre[i])
            print(self.C6_reg_loss,self.P5_reg_loss,self.P4_reg_loss)
            
                    
    
    def iou_loss(self,gt,gt_predict):
        gt_X = (gt[:,:,0]+gt[:,:,2])*(gt[:,:,1]+gt[:,:,3])
        gt_predict_X = (gt_predict[:,:,0]+gt_predict[:,:,2])*(gt_predict[:,:,1]+gt_predict[:,:,3])
        I_h = tf.minimum(gt[:,:,0],gt_predict[:,:,0])+tf.minimum(gt[:,:,2],gt_predict[:,:,2])
        I_w = tf.minimum(gt[:,:,1],gt_predict[:,:,1])+tf.minimum(gt[:,:,3],gt_predict[:,:,3])
        I = I_h*I_w
        U = gt_X*gt_predict_X
        IOU = I/U
        return IOU

    def is_in_box(self,ground_truth,feature_center,gt_predict):
        ground_truth = ground_truth/img_size
        feature_center = feature_center/img_size

        x_mask = tf.cast(feature_center[:,:,0]>ground_truth[0],dtype=tf.float32)*tf.cast(feature_center[:,:,0]<ground_truth[2],dtype=tf.float32)
        y_mask = tf.cast(feature_center[:,:,1]>ground_truth[1],dtype=tf.float32)*tf.cast(feature_center[:,:,1]<ground_truth[3],dtype=tf.float32)
        mask = x_mask*y_mask

        label_gt_l = tf.expand_dims(feature_center[:,:,0]-ground_truth[0],axis=-1)
        label_gt_r = tf.expand_dims(ground_truth[2] - feature_center[:,:,0],axis=-1)
        label_gt_t = tf.expand_dims(feature_center[:,:,1]-ground_truth[1],axis=-1)
        label_gt_b = tf.expand_dims(ground_truth[3] - feature_center[:,:,1],axis=-1)

        label_gt = tf.concat([label_gt_l,label_gt_t,label_gt_r,label_gt_b],axis=-1)
        loss = iou_loss(label_gt,gt_predict)
        return mask*loss
    
    
    
    def reg_loss(self,ground_truth,feature_center,gt_predict):
        def cal_loss(ground_truth,feature_center,gt_predict):
            stride_length = gt_predict.shape[0]
            s_area = self.img_size//stride_length
            area = (ground_truth[2]-ground_truth[0])*(ground_truth[3]-ground_truth[1])
            print(area)
            return tf.cond(tf.greater(area,s_area),
                           lambda: is_in_box(ground_truth,feature_center,gt_predict),lambda: 0)
        
        loss = tf.map_fn(lambda x:cal_loss(x,feature_center,gt_predict),elems=ground_truth)
        return loss
    
    
    def build_reg_net(self,x):
        for i in range(4):
            x = tf.layers.conv2d(x,256,3,1,padding='SAME')
            x = self.batch_relu(x)
        x = tf.layers.conv2d(x,4,1,1,padding='SAME')
        
        return x
    
    def fpn_net(self,x1,x2):
        x1 = tf.layers.conv2d(x1,x1.shape[-1],1,1,padding='SAME')
        x1 = self.batch_relu(x1)
        
        x2 = tf.layers.conv2d_transpose(x2,x1.shape[-1],3,2,padding='SAME')
        x2 = self.batch_relu(x2)
        
        return x1+x2
    
    def build_cs_net(self,x):
        for i in range(4):
            x = tf.layers.conv2d(x,256,3,1,padding='SAME')
            x = self.batch_relu(x)
        cs_net = tf.layers.conv2d(x,1,1,1,padding='SAME')
        cls_net = tf.layers.conv2d(x,self.class_num,1,1,padding='SAME')
        
        return cs_net,cls_net
    
    def resnet_block(self,x,ch,block_num,fil_size,is_first=False,scope_name=''):
        with tf.variable_scope(scope_name):
            if(is_first==True):
                x = self.bottleneck(x,ch,stride=(1,1))
            else:
                x = self.bottleneck(x,ch,fil_size=fil_size,stride=(2,2))
            for i in range(block_num-1):
                x = self.bottleneck(x,ch,stride=(1,1),scope_name=scope_name+str(i))
            return x
    
    def bottleneck(self,x,ch,stride=(2,2),fil_size=5,scope_name=''):
        with tf.variable_scope(scope_name):
            short_cut = x
            if(stride==(2,2)):
                channel = x.shape[-1]//2
                #short_cut = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                short_cut = self.blur_pooling(short_cut,5,pool_type='avg',scope_name='avg_blur0')
                short_cut = tf.layers.conv2d(short_cut,ch*4,1,1,padding='SAME',
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            elif(x.shape[-1]==64):
                 short_cut = tf.layers.conv2d(short_cut,ch*4,1,1,padding='SAME',
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
                

            x2 = self.batch_relu(x)
            if(stride==(1,1)):
                x2 = tf.layers.conv2d(x2,ch,3,stride,padding='SAME',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            else:
                x2 = self.blur_pooling(x2,3,pool_type='conv',scope_name='conv_blur0')


            x2 = self.res2net_block(x2,ch)

            x2 = self.batch_relu(x2)
            x2 = tf.layers.conv2d(x2,ch*4,1,1,padding='SAME',
                                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

            x2 = self.SE_module(x2)
            #x2 = self.cbam_module(x2)
            x = short_cut+x2
        
        return x
    
    def res2net_block(self,x,ch): 
        s = 4
        block = list()
        split_dim = int(x.shape[-1])//s
        pre_block = None
        for i in range(s):
            now_block = x[:,:,:,i*split_dim:(i+1)*split_dim]
            
            if(i>1):
                #print(pre_block,now_block)
                now_block = tf.add(now_block,pre_block)
            if(i>0):
                #now_block = self.pre_act(pre_block,trainable)
                now_block = self.batch_relu(now_block)
                now_block = tf.layers.conv2d(pre_block,split_dim,3,1,padding='SAME')
    
            block.append(now_block)
            pre_block = now_block
        return self.Concatenation(block)
    
    def Concatenation(self,layers,axis=-1) :
        return tf.concat(layers, axis=axis)
    
    def batch_relu(self,x):
        x = tf.layers.batch_normalization(x,training=self.trainable)
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
