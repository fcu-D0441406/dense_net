import tensorflow as tf
import numpy as np
import os

class Dense_net:
    
    def __init__(self,img_size,channel,class_num,k,grow_rate,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.drop_rate = 0.3
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[None,self.img_size,self.img_size,self.channel])
        self.x_loc = tf.placeholder(tf.float32,[None,4])
        
        self.anchor_box = {'0':32,'1':16,'2':8,'3':4}
        self.ratio = [0.5,1,2]
        self.anchor_num = 3
        self.iou_thresh = 0.6
        self.build_net(trainable)
        #self.upsample(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            #tf.contrib.layers.variance_scaling_initializer()
            dense1 = tf.layers.conv2d(self.x,2*self.k,5,2,padding='SAME',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            self.dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            print(self.dense1)
            self.dense2 = self.dense_block(self.dense1,6,trainable)
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,12,trainable)
            print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,16,trainable)
            print(self.dense4)
            
            '''
            dense5 = self.dense_block(dense4,16,trainable)
            print(dense5)
            '''
            
            avg_pool = tf.layers.average_pooling2d(self.dense3,2,1)
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
            
    
    def fpn_net(self):
        ch = self.dense4.shape[3]
        
        self.pre0 = tf.layers.conv2d(self.dense4,ch,1,1,padding='SAME')
        #print(self.pre0)
        
        self.pre1 = tf.add(tf.layers.conv2d_transpose(self.pre0,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense3,ch,1,1,padding='SAME'))
        self.pre1 = tf.layers.conv2d(self.pre1,ch,3,1,padding='SAME')
        #print(self.pre1)
        self.pre2 = tf.add(tf.layers.conv2d_transpose(self.pre1,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense2,ch,1,1,padding='SAME'))
        self.pre2 = tf.layers.conv2d(self.pre2,ch,3,1,padding='SAME')
        #print(self.pre2)
        self.pre3 = tf.add(tf.layers.conv2d_transpose(self.pre2,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense1,ch,1,1,padding='SAME'))
        self.pre3 = tf.layers.conv2d(self.pre3,ch,3,1,padding='SAME')
        #print(self.pre3)
    
    def rpn_net(self):
        
        self.r_net0 = tf.layers.conv2d(self.pre0,256,3,1,padding='SAME')
        #print(self.r_net0)
        self.bf_check0 = tf.layers.conv2d(self.r_net0,2*self.anchor_num,1,1,padding='SAME')
        #print(self.bf_check0)
        self.loc_check0 = tf.layers.conv2d(self.r_net0,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check0)
        
        self.r_net1 = tf.layers.conv2d(self.pre1,256,3,1,padding='SAME')
        #print(self.r_net1)
        self.bf_check1 = tf.layers.conv2d(self.r_net1,2*self.anchor_num,1,1,padding='SAME')
        #print(self.bf_check1)
        self.loc_check1 = tf.layers.conv2d(self.r_net1,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check1)
        
        self.r_net2 = tf.layers.conv2d(self.pre2,256,3,1,padding='SAME')
        #print(self.r_net2)
        self.bf_check2 = tf.layers.conv2d(self.r_net2,2*self.anchor_num,1,1,padding='SAME')
        #print(self.bf_check2)
        self.loc_check2 = tf.layers.conv2d(self.r_net2,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check2)
        
        self.r_net3 = tf.layers.conv2d(self.pre3,256,3,1,padding='SAME')
        #print(self.r_net3)
        self.bf_check3 = tf.layers.conv2d(self.r_net3,2*self.anchor_num,1,1,padding='SAME')
        #print(self.bf_check3)
        self.loc_check3 = tf.layers.conv2d(self.r_net3,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check3)
    
    def score_softmax(self):
        self.bf_check0 = tf.reshape(self.bf_check0,(-1,
                                                    self.bf_check0.shape[1],
                                                    self.bf_check0.shape[2]*self.anchor_num,
                                                    2))
        self.bf_check0 = tf.nn.softmax(self.bf_check0)
        self.bf_check0 = tf.reshape(self.bf_check0,(-1,
                                                    self.bf_check0.shape[1],
                                                    self.bf_check0.shape[2],
                                                    2*self.anchor_num))
        
        self.bf_check1 = tf.reshape(self.bf_check1,(-1,
                                                    self.bf_check1.shape[1],
                                                    self.bf_check1.shape[3]*self.anchor_num,
                                                    2))
        self.bf_check1 = tf.nn.softmax(self.bf_check1)
        self.bf_check1 = tf.reshape(self.bf_check1,(-1,
                                                    self.bf_check1.shape[1],
                                                    self.bf_check1.shape[3],
                                                    2*self.anchor_num))
        
        self.bf_check2 = tf.reshape(self.bf_check2,(-1,
                                                    self.bf_check2.shape[1],
                                                    self.bf_check2.shape[3]*self.anchor_num,
                                                    2))
        self.bf_check2 = tf.nn.softmax(self.bf_check2)
        self.bf_check2 = tf.reshape(self.bf_check2,(-1,
                                                    self.bf_check2.shape[1],
                                                    self.bf_check2.shape[3],
                                                    2*self.anchor_num))
        
        self.bf_check3 = tf.reshape(self.bf_check3,(-1,
                                                    self.bf_check3.shape[1],
                                                    self.bf_check3.shape[3]*self.anchor_num,
                                                    2))
        self.bf_check3 = tf.nn.softmax(self.bf_check3)
        self.bf_check3 = tf.reshape(self.bf_check3,(-1,
                                                    self.bf_check3.shape[1],
                                                    self.bf_check3.shape[3],
                                                    2*self.anchor_num))
        
    def select_anchor(self):
        box = []
        box = self.cal_fg_anchor(self.bf_check0,box,'0')
        box = self.cal_fg_anchor(self.bf_check1,box,'1')
        box = self.cal_fg_anchor(self.bf_check2,box,'2')
        box = self.cal_fg_anchor(self.bf_check3,box,'3')
        print(len(box))
    
    def cal_fg_anchor(self,bf_check,box,p):
        
        def cal_pre_nms(i,j):
            width = self.anchor_box[p]*self.ratio[a]
            height = self.anchor_box[p]*self.ratio[a]
            y1 = (i+1)*self.anchor_box[p]*self.ratio[a]-(height/2)
            x1 = (j+1)*self.anchor_box[p]*self.ratio[a]-(width/2)
            y2 = (i+1)*self.anchor_box[p]*self.ratio[a]+(height/2)
            x2 = (j+1)*self.anchor_box[p]*self.ratio[a]+(width/2)
            if(x1<0 or y1<0 or x2>224 or y2>224):
                return None
            if(self.cal_nms(x1,y1,x2,y2)):
                loc = [x1,y1,x2,y2]
                return loc
        def nothing():
            return None
        
        for i in range(bf_check.shape[1]):
            for j in range(bf_check.shape[2]):
                for a in range(self.anchor_num):
                    result = tf.cond(bf_check[0][i][j][a]>=0.8,lambda:cal_pre_nms(i,j),
                                     lambda:nothing())
                    if(result!=None):
                        box.append(result)
                    
                    if(bf_check[i][j][a]>=0.8):
                        width = self.anchor_box[p]*self.ratio[a]
                        height = self.anchor_box[p]*self.ratio[a]
                        y1 = (i+1)*self.anchor_box[p]*self.ratio[a]-(height/2)
                        x1 = (j+1)*self.anchor_box[p]*self.ratio[a]-(width/2)
                        y2 = (i+1)*self.anchor_box[p]*self.ratio[a]+(height/2)
                        x2 = (j+1)*self.anchor_box[p]*self.ratio[a]+(width/2)
                        if(x1<0 or y1<0 or x2>224 or y2>224):
                            continue
                        if(self.cal_nms(x1,y1,x2,y2)):
                            loc = [x1,y1,x2,y2]
                            box.append(loc)
                    
        return box
        
    
    
                        
    
    def cal_nms(self,x1,y1,x2,y2):
        pre_area = (x2-x1)*(y2-y1)
        for i in range(1):
            l_x1,l_y1,l_x2,l_y2 = self.x_loc[i][0],self.x_loc[i][1],self.x_loc[i][2],self.x_loc[i][3]
            label_area = (l_x2-l_x1)*(l_y2-l_y1)
            xx1 = tf.maximum(x1,l_x1)
            yy1 = tf.maximum(y1,l_y1)
            xx2 = tf.maximum(x2,l_x2)
            yy2 = tf.maximum(y2,l_y2)
            n_width = xx1-xx2
            n_height = yy1-yy2
            if(n_width<0 or n_height<0):
                break
            else:
                n_area = n_width*n_height
                nms = n_area/label_area
                #nms = n_area/(label_area+pre_area-n_area)
            if(nms>=self.iou_thresh):
                return True
        return False
            
        
    
    def dense_block(self,x,block_num,trainable):
        for i in range(block_num):
            if(i==0):
                x = self.bottleneck(x,trainable)
                #print(x)
            else:
                con_x = self.bottleneck(x,trainable)
                x = tf.concat([x,con_x],axis=3)
                #print(x)
        ch = (self.k*block_num)*self.grow_rate
        #print(ch)
        
        x = tf.layers.conv2d(x,ch,1,1,padding='SAME')
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x,self.drop_rate)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        return x
    
    def bottleneck(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k*4,1,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        x = tf.layers.dropout(x,self.drop_rate)
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,3,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        x = tf.layers.dropout(x,self.drop_rate)
        return x
    
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

if(__name__=='__main__'):
    ds = Dense_net(224,3,10,12,0.5)
    ds.fpn_net()
    ds.rpn_net()
    print(ds.bf_check0)
    ds.score_softmax()
    print(ds.bf_check0)
    ds.select_anchor()
    #print(resnet.predict)

