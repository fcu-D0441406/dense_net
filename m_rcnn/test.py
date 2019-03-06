import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import dense_net

anchor_num = 4
a_num = 1
ratio_num = 4
stride=[7,14,28,56]
anchor=[64,32,16,8]
ratio = np.array([[1,1],[1,2],[2,1],[2,2]])
ckpts = './checkpoint1_dir'
adam_meta = './checkpoint1_dir/MyModel'




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
    
    real_fg = tf.placeholder(tf.float32,[None,16660])
    is_real_fg = tf.placeholder(tf.float32,[None,16660])
    fg_num = tf.placeholder(tf.float32)
    
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        ########  save
        var_list = tf.trainable_variables()
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
            ba_x = i%67
            img = cv2.imread(train_data[ba_x][0])
            r_img = img[np.newaxis,:,:,:]
            pfi,pfs,p_box = sess.run([pre_fg_index,pre_fg_score,pre_box_dx],feed_dict={x:r_img})
            draw_img(img,pfi,p_box,pfs)
                
        
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

def draw_img(img,pfi,p_box,pfs):
    p0 = 0
    p1 = 7*7*anchor_num
    p2 = p1+14*14*anchor_num
    p3 = p2+28*28*anchor_num
    p4 = p3+56*56*anchor_num
    all_p = [p0,p1,p2,p3,p4]
    loc = []
    for pfi_num in range(int(pfi.shape[0])):
        for p_index in range(2000):
            for j in range(len(all_p)):
                if(pfi[pfi_num][p_index]>=all_p[j] and pfi[pfi_num][p_index]<all_p[j+1]):
                    #print(pfi[pfi_num][p_index],pfs[pfi_num][p_index])
                    now = pfi[pfi_num][p_index]-all_p[j]
                    now_c = now%anchor_num
                    now = int(now/anchor_num)
                    now_y = int(now/stride[j])
                    now_x = now%stride[j]
                    height = ratio[now_c][0]*anchor[j]
                    width = ratio[now_c][1]*anchor[j]
                    
                    cx = now_x*anchor[j] + width*p_box[pfi_num][pfi[pfi_num][p_index]][0]
                    cy = now_y*anchor[j] + height*p_box[pfi_num][pfi[pfi_num][p_index]][1]
                    c_width = width*float(np.exp(p_box[pfi_num][pfi[pfi_num][p_index]][2]))
                    c_height = height*float(np.exp(p_box[pfi_num][pfi[pfi_num][p_index]][3]))
                    
                    x1 = cx-(c_width/2)
                    y1 = cy-(c_height/2)
                    x2 = x1+c_width
                    y2 = y1+c_height
                    if(x1<0 or y1<0 or x2>224 or y2>224):
                        continue
                    #print(now_y,now_x)
                    loc.append([x1,y1,x2,y2,pfs[pfi_num][p_index]])
                    #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    loc = np.array(loc)
    loc = py_cpu_nms(loc, 0.2)
    for pfi_num in range(int(pfi.shape[0])):
        for p_index in loc:
            for j in range(len(all_p)):
                if(pfi[pfi_num][p_index]>=all_p[j] and pfi[pfi_num][p_index]<all_p[j+1]):
                    #print(pfi[pfi_num][p_index],pfs[pfi_num][p_index])
                    now = pfi[pfi_num][p_index]-all_p[j]
                    now_c = now%anchor_num
                    now = int(now/anchor_num)
                    now_y = int(now/stride[j])
                    now_x = now%stride[j]
                    height = ratio[now_c][0]*anchor[j]
                    width = ratio[now_c][1]*anchor[j]
                    
                    cx = now_x*anchor[j] + width*p_box[pfi_num][pfi[pfi_num][p_index]][0]
                    cy = now_y*anchor[j] + height*p_box[pfi_num][pfi[pfi_num][p_index]][1]
                    c_width = width*float(np.exp(p_box[pfi_num][pfi[pfi_num][p_index]][2]))
                    c_height = height*float(np.exp(p_box[pfi_num][pfi[pfi_num][p_index]][3]))
                    
                    x1 = cx-(c_width/2)
                    y1 = cy-(c_height/2)
                    x2 = x1+c_width
                    y2 = y1+c_height
                    if(x1<0 or y1<0 or x2>224 or y2>224):
                        continue
                    #print(now_y,now_x)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    show_img(img)

def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

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

train()