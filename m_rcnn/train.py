import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import dense_net

anchor_num = 3
a_num = 1
ratio_num = 3
stride=[7,14,28,56]
anchor=[96,64,48,32]
ratio = np.array([[1,1],[1,2],[2,1]])
ckpts = './checkpoint2_dir'
adam_meta = './checkpoint2_dir/MyModel'


def cal_IOU(Reframe,GTframe):
    img = cv2.imread('test.jpg')
    if(Reframe[0]<0 or Reframe[1]<0 or Reframe[2]>224 or Reframe[3]>224):
        return 0,0.0
    elif(Reframe[0]>=Reframe[2] or Reframe[1]>=Reframe[3]):
        return 0,0.0
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)
    IOU = 0
    if width <=0 or height <= 0:
        result = 0
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        IOU = Area*1./(Area1+Area2-Area)
        if(IOU>=0.5):
            print(IOU)
            '''
            cv2.rectangle(img, (int(Reframe[0]), int(Reframe[1])), (int(Reframe[2]), int(Reframe[3])), (0, 255, 0), 2)
            show_img(img)
            '''
            result = 1
        elif(IOU>=0.3):
            result = 0
        else:
            result = -1
    return result,IOU

def select_anchor(pfi,pfs,p_box,r_box):
    p0 = 0
    p1 = 7*7*anchor_num
    p2 = p1+14*14*anchor_num
    p3 = p2+28*28*anchor_num
    p4 = p3+56*56*anchor_num
    all_p = [p0,p1,p2,p3,p4]
    real_fg_score = np.zeros(shape=pfs.shape)
    is_anchor = np.zeros(shape=pfs.shape)
    is_anchor_num = 0
    best_dx_box = np.zeros([4])
    dx_box = np.zeros(shape=p_box.shape)
    is_dx_box = np.zeros(shape=p_box.shape)
    
    pos = 0
    neg = 0
    neg_record = np.zeros([256,2],dtype=np.int)

    positive = False
    max_nms = [0,0,0]
    
    '''
    i: 真實box個數
    '''
    
    for pfi_num in range(int(pfi.shape[0])):
        now_p = -1
        for i in range(int(r_box.shape[0])):
            for p_index in pfi[pfi_num]:
                now_p+=1
                for j in range(len(all_p)):
                    #print(p_index,j)
                    if(p_index>=all_p[j] and p_index<all_p[j+1]):
                        now = p_index-all_p[j]
                        now_c = now%anchor_num
                        now = int(now/anchor_num)
                        now_y = int(now/stride[j])
                        now_x = now%stride[j]
                        height = ratio[now_c][0]*anchor[j]
                        width = ratio[now_c][1]*anchor[j]
                        
                        r_cx = (r_box[i][0]+r_box[i][2])/2
                        r_cy = (r_box[i][1]+r_box[i][3])/2
                        r_width = (r_box[i][2]-r_box[i][0])
                        r_height = (r_box[i][3]-r_box[i][1])
                        
                        try:
                            cx = now_x*anchor[j] + width*p_box[pfi_num][now_p][0]
                            cy = now_y*anchor[j] + height*p_box[pfi_num][now_p][1]
                            c_width = width*float(np.exp(p_box[pfi_num][now_p][2]))
                            c_height = height*float(np.exp(p_box[pfi_num][now_p][3]))
                        except:
                            continue
                        x1 = cx-(c_width/2)
                        y1 = cy-(c_height/2)
                        x2 = x1+c_width
                        y2 = y1+c_height
                        
                        #####
                        '''
                        print(x1,y1,x2,y2)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        show_img(img)
                        '''
                        #####
                        result,nms = cal_IOU([x1,y1,x2,y2],r_box[i])
                        if(result==1 and pos<=128):
                            #print(now_p)
                            positive=True
                            real_fg_score[pfi_num][now_p] = 1
                            is_anchor[pfi_num][now_p] = 20
                            is_dx_box[pfi_num][now_p][:] = 1
                            is_anchor_num+=1
                            dx_box[pfi_num][now_p][0] = (r_cx-now_x*anchor[j])/width
                            dx_box[pfi_num][now_p][1] = (r_cy-now_y*anchor[j])/height
                            dx_box[pfi_num][now_p][2] = np.log(r_width/width)
                            dx_box[pfi_num][now_p][3] = np.log(r_height/height)
                            #print(dx_box[pfi_num][p_index])
                            pos+=1
                        elif(result==-1 and neg<256):
                            neg_record[neg] = pfi_num,now_p
                            neg+=1
                        elif(result==0 and nms>=0.3):
                            if(nms>max_nms[0] and positive==False):
                                max_nms[0] = nms
                                max_nms[1] = pfi_num
                                max_nms[2] = now_p
                                best_dx_box[0] = (r_cx-now_x*anchor[j])/width
                                best_dx_box[1] = (r_cy-now_y*anchor[j])/height
                                best_dx_box[2] = np.log(r_width/width)
                                best_dx_box[3] = np.log(r_height/height)
                            '''
                            if(mid<256):
                                mid_record[mid] = pfi_num,p_index
                                mid+=1
                            '''
        if(positive==False):
            if(max_nms[0]>=0.3):
                print(max_nms[0])
                real_fg_score[max_nms[1]][max_nms[2]] = 1
                is_anchor[max_nms[1]][max_nms[2]] = 1
                dx_box[max_nms[1]][max_nms[2]] = best_dx_box
                is_dx_box[max_nms[1]][max_nms[2]][:] = 1
                pos+=1
                #print(best_dx_box)
                is_anchor_num+=1
        neg = 0
        mid = 0
        if(pos==0):
            pos=1  
        all_limit = 256
        neg_weight = 1
        #other_limit = 64
        if(is_anchor_num<all_limit):
            for j in range(neg_record.shape[0]):
                if(is_anchor_num<all_limit):
                    pn,pi = neg_record[j]
                    #print(pfi_num,now_p)
                    real_fg_score[pn][pi] = 0
                    is_anchor[pn][pi] = 1*neg_weight
                    is_anchor_num+=1
                    neg+=1
                else:
                    break
          
    print(is_anchor_num,pos,mid,neg)
    
    return real_fg_score,is_anchor,is_anchor_num,dx_box,pos,is_dx_box

def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cal_l1_loss(real_box_dx,pre_box,is_real_box,ll1,ll2,ll3):
    j1 = tf.add(real_box_dx,(-1)*pre_box)
    j2 = tf.cast(tf.less(tf.abs(j1),ll1),tf.float32)
    j3 = is_real_box*((j2)*(ll2*j1*j1)+(1.0-j2)*(tf.abs(j1)-ll3))
    j4 = tf.reduce_sum(j3)
    return j4,j3

def save(sess,global_step):
    ########  save
    var_list = tf.trainable_variables()
    if global_step is not None:
        var_list.append(global_step)
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list,max_to_keep=5)
    ########
    if tf.train.latest_checkpoint(ckpts) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(ckpts))
    else:
        assert 'can not find checkpoint folder path!'

def train():
    
    train_data = read_xml('./outputs')
    '''
    img = cv2.imread('test.jpg')
    r_img = np.reshape(img,(-1,224,224,3))
    r_box = np.reshape(np.array([78,54,148,173],dtype=np.float),(-1,4))
    '''
    
    ds = dense_net.Dense_net(224,3,10,24,0.5)
    x = ds.x
    
    pre_fg_index,pre_fg_score = ds.best_index,ds.best_fg_score
    pre_box_dx = ds.all_box
    
    real_fg = tf.placeholder(tf.float32,[None,None])
    is_real_fg = tf.placeholder(tf.float32,[None,None])
    fg_num = tf.placeholder(tf.float32)
    
    real_box_dx = tf.placeholder(tf.float32,[None,None,4])
    is_real_box_dx = tf.placeholder(tf.float32,[None,None,4])
    box_num = tf.placeholder(tf.float32)
    
    
    
    #fg_loss = tf.reduce_sum(tf.multiply(tf.squared_difference(real_fg,pre_fg_score),is_real_fg))/fg_num
    fg_loss = (-1)*tf.reduce_sum(is_real_fg*(real_fg * tf.log(tf.clip_by_value(pre_fg_score,1e-10,1.0))+
               (1-real_fg) * tf.log(tf.clip_by_value(1-pre_fg_score,1e-10,1.0))))
    fg_loss = fg_loss/fg_num
    #fg_loss = tf.reduce_sum(tf.squared_difference(real_fg,pre_fg_score)*is_real_fg)/fg_num
    #fg_loss,_ = cal_l1_loss(real_fg,pre_fg_score,is_real_fg,0.3,1.0,0.0)
    #fg_loss = fg_loss/fg_num
    #fg_loss = tf.reduce_sum(tf.abs(real_fg-pre_fg_score)*is_real_fg)/fg_num
    box_loss,j1 = cal_l1_loss(real_box_dx,pre_box_dx,is_real_box_dx,1.0,0.5,0.5)
    box_loss = box_loss/box_num
    box_loss = tf.clip_by_value(box_loss,1e-6,box_loss)
    total_loss = fg_loss+box_loss
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    #opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')
    with tf.name_scope('opt'):
        #opt = tf.train.AdamOptimizer(1e-3,name='optimizer')
        #opt = tf.train.GradientDescentOptimizer(5e-3,name='optimizer')
        opt = tf.train.MomentumOptimizer(1e-3,0.9,name='optimizer')
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(total_loss)
            train_op = opt.apply_gradients(grads, global_step=global_step)
        #tf.summary.scalar('opti', opt)
    #train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss)
    #train_op = tf.train.AdamOptimizer(5e-3).minimize(box_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        ########  save
        var_list = tf.trainable_variables()
        if global_step is not None:
            var_list.append(global_step)
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list,max_to_keep=5)
        ########
        
        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'
        
        for i in range(10000):
            print('-----')
            ba_x = i%96
            img = cv2.imread(train_data[ba_x][0])
            r_img = img[np.newaxis,:,:,:]
            r_box = np.reshape(np.array(train_data[ba_x][2:]),(-1,4))
            
            
            pfi,pfs,p_box = sess.run([pre_fg_index,pre_fg_score,pre_box_dx],feed_dict={x:r_img})
            real_fg_score,is_anchor,is_anchor_num,r_dx_box,box_n,is_dx_box = select_anchor(pfi,pfs,p_box,r_box)
            print(is_anchor_num)
            #print(r_dx_box.shape)
            fg_ls,bx_ls,t_ls,tttt,_ = sess.run([fg_loss,box_loss,total_loss,pre_fg_score,train_op],
                                     feed_dict={x:r_img,real_fg:real_fg_score,is_real_fg:is_anchor,fg_num:is_anchor_num,
                                                real_box_dx:r_dx_box,box_num:box_n,is_real_box_dx:is_dx_box})
            
            for j in range(real_fg_score.shape[1]):
                if(real_fg_score[0][j]==1):
                    print(j,real_fg_score[0][j],pfs[0][j])
            
            if(i%100==0 and i!=0):
                #draw_img(img,pfi,p_box)
                saver.save(sess,adam_meta,global_step=i) 
            
            print(fg_ls,bx_ls)
            print(tttt[0][0],tttt[0][-1])
            print('--------')
                
        
def read_xml(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        #print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,
                     #int(root.find('size')[0].text),
                     #int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            #print(value)
            xml_list.append(value)
    return xml_list

def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  
    order = scores.argsort()[::-1]  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    keep = []  
    while order.size > 0:  
 
        i = order[0]  
        keep.append(i)  

        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  

        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
  
        inds = np.where(ovr <= thresh)[0]  

        order = order[inds + 1]  
  
    return keep

if(__name__=='__main__'):
    train()