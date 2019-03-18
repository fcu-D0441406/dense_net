import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import dense_net

import anchor_op

anchor_num = 3
a_num = 1
ratio_num = 3
stride=np.array([7,14,28,56])
anchor_size=np.array([64,32,16,8])
ratio = np.array([[1,1],[1,2],[2,1]])
ckpts = './checkpoint2_dir'
adam_meta = './checkpoint2_dir/MyModel'
    

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

def train():
    train_data = read_xml('./outputs')
    '''
    img = cv2.imread('test.jpg')
    r_img = np.reshape(img,(-1,224,224,3))
    r_box = np.reshape(np.array([[78,54,148,173],[28,12,64,99]],dtype=np.float),(-1,4))
    '''
    ds = dense_net.Dense_net(224,3,10,24,0.5)
    x = ds.x
    
    pre_fg,pre_fg_score = ds.all_fg,ds.all_fg_score
    pre_box_dx = ds.all_box
    
    sort_fg_index,sort_fg_score = ds.sort_fg_score()
    
    real_fg = tf.placeholder(tf.float32,[None,None,2])
    is_real_fg = tf.placeholder(tf.float32,[None,None])
    fg_num = tf.placeholder(tf.float32)
    
    real_box_dx = tf.placeholder(tf.float32,[None,None,4])
    is_real_box_dx = tf.placeholder(tf.float32,[None,None,4])
    box_num = tf.placeholder(tf.float32)
    
    #fg_loss = tf.reduce_sum(tf.multiply(tf.squared_difference(real_fg,pre_fg_score),is_real_fg))/fg_num
    #fg_loss = (-1)*tf.reduce_sum(is_real_fg*(real_fg * tf.log(tf.clip_by_value(pre_fg_score,1e-10,1.0))+
    #           (1-real_fg) * tf.log(tf.clip_by_value(1-pre_fg_score,1e-10,1.0))))
    #fg_loss = fg_loss/fg_num
    #fg_loss = tf.reduce_sum(tf.squared_difference(real_fg,pre_fg_score)*is_real_fg)/fg_num
    #fg_loss,_ = cal_l1_loss(real_fg,pre_fg_score,is_real_fg,0.3,1.0,0.0)
    #fg_loss = fg_loss/fg_num
    #print(pre_fg.shape)
    fg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=tf.argmax(real_fg,axis=2),logits=pre_fg)

    fg_loss = tf.reduce_sum(tf.multiply(fg_loss,is_real_fg),axis=1)
    fg_loss = fg_loss[0]/fg_num
    box_loss,j1 = cal_l1_loss(real_box_dx,pre_box_dx,is_real_box_dx,1.0,0.5,0.5)
    box_loss = (box_loss/box_num)*5
    box_loss = tf.clip_by_value(box_loss,1e-10,box_loss)
    rpn_total_loss = fg_loss+box_loss
    
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('opt'):
        #opt = tf.train.AdamOptimizer(1e-3,name='optimizer')
        #opt = tf.train.GradientDescentOptimizer(5e-3,name='optimizer')
        opt = tf.train.MomentumOptimizer(1e-4,0.9,name='optimizer')
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(rpn_total_loss)
            train_op = opt.apply_gradients(grads, global_step=global_step)
        #tf.summary.scalar('opti', opt)
    
    anchor_ut = anchor_op.anchor_util(stride,anchor_size)
    
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
            #saver = tf.train.import_meta_graph('./checkpoint1_dir/MyModel-3250.meta')
        else:
            assert 'can not find checkpoint folder path!'
        
        
        for i in range(100010):
            print('-----')
            
            ba_x = i%96
            img = cv2.imread(train_data[ba_x][0])
            r_img = img[np.newaxis,:,:,:]
            r_box = np.reshape(np.array(train_data[ba_x][2:]),(-1,4))
            #print(r_img.shape)
            
            pf_value,pf_score,dx_box,sort_id,sort_score,td = sess.run([pre_fg,pre_fg_score,pre_box_dx,
                                                 sort_fg_index,sort_fg_score,ds.dense4],feed_dict={x:r_img})
    
            dx_box_loc = anchor_ut.get_dx_box(dx_box[0])
            
            fg_bg_check,real_fg_bg_value,box_index,is_anchor = anchor_ut.collect_fg_bg(dx_box_loc,r_box)
            
            dx_anchor_value,is_dx_box,is_box = anchor_ut.collect_anchor_dx(dx_box,box_index,real_fg_bg_value,r_box)
                          
            
            fg_ls,box_ls,fg_rank,_= sess.run([fg_loss,box_loss,sort_fg_index,train_op],feed_dict={x:r_img,
                                               real_box_dx:dx_anchor_value,is_real_box_dx:is_dx_box,box_num:is_box,
                                               real_fg:real_fg_bg_value,is_real_fg:fg_bg_check,fg_num:is_anchor
                                               })
                    
            if(i%1==0):
                for j in range(pf_score.shape[1]):
                    if(real_fg_bg_value[0][j][0]==1):
                        rank = 0
                        for k in range(fg_rank.shape[0]):
                            if(fg_rank[k]==j):
                                rank = k
                        #print('high',pf_score[0][fg_rank[0]])
                        #print('low',pf_score[0][fg_rank[-1]])
                        print('fg_score',pf_score[0][j],'rank',rank,' id',j)
                        print(fg_ls,box_ls)
                        
            if(i%50==0 and i!=0):
                saver.save(sess,adam_meta,global_step=i) 
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


if(__name__=='__main__'):
    train()