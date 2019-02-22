import tensorflow as tf
import numpy as np
import os
import cv2

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
            #print(self.dense1)
            self.dense2 = self.dense_block(self.dense1,6,trainable)
            #print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,12,trainable)
            #print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,16,trainable)
            #print(self.dense4)
            
            '''
            dense5 = self.dense_block(dense4,16,trainable)
            print(dense5)
            '''
            
            avg_pool = tf.layers.average_pooling2d(self.dense3,2,1)
            #print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            #print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            #print(self.prediction)
            
    
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
        
        self.r_net0 = tf.layers.conv2d(self.pre0,512,3,1,padding='SAME')
        #print(self.r_net0)
        self.bf_check0 = tf.layers.conv2d(self.r_net0,2*self.anchor_num,1,1,padding='SAME')
        print(self.bf_check0)
        self.loc_check0 = tf.layers.conv2d(self.r_net0,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check0)
        
        self.r_net1 = tf.layers.conv2d(self.pre1,512,3,1,padding='SAME')
        #print(self.r_net1)
        self.bf_check1 = tf.layers.conv2d(self.r_net1,2*self.anchor_num,1,1,padding='SAME')
        print(self.bf_check1)
        self.loc_check1 = tf.layers.conv2d(self.r_net1,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check1)
        
        self.r_net2 = tf.layers.conv2d(self.pre2,512,3,1,padding='SAME')
        #print(self.r_net2)
        self.bf_check2 = tf.layers.conv2d(self.r_net2,2*self.anchor_num,1,1,padding='SAME')
        print(self.bf_check2)
        self.loc_check2 = tf.layers.conv2d(self.r_net2,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check2)
        
        self.r_net3 = tf.layers.conv2d(self.pre3,512,3,1,padding='SAME')
        #print(self.r_net3)
        self.bf_check3 = tf.layers.conv2d(self.r_net3,2*self.anchor_num,1,1,padding='SAME')
        print(self.bf_check3)
        self.loc_check3 = tf.layers.conv2d(self.r_net3,4*self.anchor_num,1,1,padding='SAME')
        #print(self.loc_check3)
    
    def score_softmax(self):
        init_shape = self.bf_check0.shape[2]
        self.bf_check0 = tf.reshape(self.bf_check0,(-1,
                                                    self.bf_check0.shape[1],
                                                    self.bf_check0.shape[2]*self.anchor_num,
                                                    2))
        #print(self.bf_check0)
        self.bf_check0 = tf.nn.softmax(self.bf_check0)
        self.bf_check0 = tf.reshape(self.bf_check0,(-1,
                                                    self.bf_check0.shape[1],
                                                    init_shape,
                                                    2*self.anchor_num))
        #print(self.bf_check0)
        
        init_shape = self.bf_check1.shape[2]
        self.bf_check1 = tf.reshape(self.bf_check1,(-1,
                                                    self.bf_check1.shape[1],
                                                    self.bf_check1.shape[2]*self.anchor_num,
                                                    2))
        self.bf_check1 = tf.nn.softmax(self.bf_check1)
        self.bf_check1 = tf.reshape(self.bf_check1,(-1,
                                                    self.bf_check1.shape[1],
                                                    init_shape,
                                                    2*self.anchor_num))
        
        init_shape = self.bf_check2.shape[2]
        self.bf_check2 = tf.reshape(self.bf_check2,(-1,
                                                    self.bf_check2.shape[1],
                                                    self.bf_check2.shape[2]*self.anchor_num,
                                                    2))
        self.bf_check2 = tf.nn.softmax(self.bf_check2)
        self.bf_check2 = tf.reshape(self.bf_check2,(-1,
                                                    self.bf_check2.shape[1],
                                                    init_shape,
                                                    2*self.anchor_num))
        
        init_shape = self.bf_check3.shape[2]
        self.bf_check3 = tf.reshape(self.bf_check3,(-1,
                                                    self.bf_check3.shape[1],
                                                    self.bf_check3.shape[2]*self.anchor_num,
                                                    2))
        self.bf_check3 = tf.nn.softmax(self.bf_check3)
        self.bf_check3 = tf.reshape(self.bf_check3,(-1,
                                                    self.bf_check3.shape[1],
                                                    init_shape,
                                                    2*self.anchor_num))
    
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



img = cv2.imread('test.jpg')

def cal_nms(x1,y1,x2,y2,label_loc):
    #print(label_loc.shape)
    pre_area = (x2-x1)*(y2-y1)
    nms = 0
    result = -1
    for i in range(label_loc.shape[0]):
        l_x1,l_y1,l_x2,l_y2 = label_loc[i][:]
        #print(l_x1,l_y1,l_x2,l_y2)
        label_area = (l_x2-l_x1)*(l_y2-l_y1)
        xx1 = max(x1,l_x1)
        yy1 = max(y1,l_y1)
        xx2 = min(x2,l_x2)
        yy2 = min(y2,l_y2)
        n_width = xx2-xx1
        n_height = yy2-yy1
        if(n_width<0 or n_height<0):
            pass
        else:
            n_area = n_width*n_height
            nms = n_area/label_area
            if(nms>=0.7):
                return 1
            elif(nms>=0.3):
                result = 0
            else:
                pass
    return result,nms

def cal_fg_anchor(bf_check,loc,p,label_loc):
    anchor_box = {'0':64,'1':32,'2':16,'3':8}
    ratio = [[1,1],[1,2],[2,1]]
    anchor_num = 3
    r_fg = np.zeros(shape=bf_check.shape)
    nms = np.zeros([bf_check.shape[0],bf_check.shape[1],anchor_num])
    positive = False
    for i in range(bf_check.shape[0]):
        for j in range(bf_check.shape[1]):
            for a in range(anchor_num):
                if(bf_check[i][j][a]>=0.5):
                    #print('hi')
                    height = anchor_box[p]*ratio[a][0]
                    width = anchor_box[p]*ratio[a][1]
                    c_x = (j+1)*anchor_box[p]*ratio[a][1]+width*loc[i][j][a*4]
                    c_y = (i+1)*anchor_box[p]*ratio[a][0]+height*loc[i][j][a*4+1]
                    #print(loc[i][j][a*4+2])
                    c_width = width*float(np.exp(int(loc[i][j][a*4+2])))
                    c_height = height*float(np.exp(int(loc[i][j][a*4+3])))
                    
                    x1 = c_x-(c_width/2)
                    y1 = c_y-(c_height/2)
                    x2 = x1+c_width
                    y2 = y1+c_height

                    if(x1<0 or y1<0 or x2>224 or y2>224):
                        continue
                    else:
                        result,n = cal_nms(x1,y1,x2,y2,label_loc)
                        nms[i][j][a] = n
                        if(result==1):
                            r_fg[i][j][a*2] = 1
                            positive = True
                        elif(result==-1):
                            r_fg[i][j][a*2+1] = 1
    if(positive==False):
        z = 0
        for i in range(nms.shape[0]):
            for j in range(nms.shape[1]):
                for a in range(anchor_num):
                    if(nms[i][j][a]>=z):
                        z = nms[i][j][a]
                        
        for i in range(nms.shape[0]):
            for j in range(nms.shape[1]):
                for a in range(anchor_num):
                    if(nms[i][j][a]==z):
                        r_fg[i][j][a*2] = 1
                        r_fg[i][j][a*2+1] = 0
                
    return r_fg

def rpn_fg_loss(fg,r_fg,r_num):
    positive_num = 0
    for i in range(r_fg.shape[0]):
        for j in range(r_fg.shape[1]):
            for a in range(3):
                

def train_rpn(sess,net):
    x = ds.x
    fg_check0,fg_check1,fg_check2,fg_check3 = ds.bf_check0,ds.bf_check1,ds.bf_check2,ds.bf_check3
    loc_check0,loc_check1,loc_check2,loc_check3 = ds.loc_check0,ds.loc_check1,ds.loc_check2,ds.loc_check3
    
    fg0,fg1,fg2,fg3 = sess.run(
                    [fg_check0,fg_check1,fg_check2,fg_check3],feed_dict={x:input_x})
    fg0,fg1,fg2,fg3 = fg0[0],fg1[0],fg2[0],fg3[0]
    loc0,loc1,loc2,loc3 = sess.run(
                    [loc_check0,loc_check1,loc_check2,loc_check3],feed_dict={x:input_x})
    loc0,loc1,loc2,loc3 = loc0[0],loc1[0],loc2[0],loc3[0]
    
    r_fg0 = cal_fg_anchor(fg0,loc0,'0',label_loc)
    r_fg1 = cal_fg_anchor(fg1,loc1,'1',label_loc)
    r_fg2 = cal_fg_anchor(fg2,loc2,'2',label_loc)
    r_fg3 = cal_fg_anchor(fg3,loc3,'3',label_loc)
    #print(r_fg0.shape,r_fg1.shape,r_fg2.shape,r_fg3.shape)
    
    loss = 0.0
    
    
    
def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if(__name__=='__main__'):
    img = cv2.imread('test.jpg')
    input_x = np.reshape(img,(-1,224,224,3))
    label_loc = np.reshape(np.array([64,83,155,140],dtype=np.float),(-1,4))
    
    ds = Dense_net(224,3,10,12,0.5)
    x = ds.x
    ds.fpn_net()
    ds.rpn_net()
    ds.score_softmax()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fg_loss = train_rpn(sess,ds)
        
        
        
        #print(loc0)
        #convert_bounding_box()
        '''
        print(anchor_data.shape)
        for i in range(anchor_data.shape[0]):
            cv2.rectangle(img,(int(anchor_data[i][0]),int(anchor_data[i][1])),(int(anchor_data[i][2]),int(anchor_data[i][3])),
                          (55,255,155),1)
        show_img(img)
        '''

        
