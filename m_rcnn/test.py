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
ckpts = './checkpoint1_dir'
adam_meta = './checkpoint1_dir/MyModel'

'''
id 25
id 26
id 28
id 45
id 46
id 47
id 49
id 50
id 67
id 68
id 70
[  46   47   25 ... 1042 1126 1127]
'''


def train():
    
    train_data = read_xml('./outputs')
    '''
    img = cv2.imread('test.jpg')
    r_img = np.reshape(img,(-1,224,224,3))
    r_box = np.reshape(np.array([78,54,148,173],dtype=np.float),(-1,4))
    '''
    anchor_ut = anchor_op.anchor_util(stride,anchor_size)
    
    ds = dense_net.Dense_net(224,3,10,24,0.5,False)
    x = ds.x
    pre_box_dx = ds.all_box
    pre_fg_index,pre_fg_score = ds.sort_fg_score()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'
        
        for i in range(97):
            print('-----')
            '''
            ba_x = i%97
            img = cv2.imread(train_data[ba_x][0])
            r_img = img[np.newaxis,:,:,:]
            '''
            img = cv2.imread('./test2.jpg')
            img = cv2.resize(img,(224,224))
            r_img = img[np.newaxis,:,:,:]

            pfi,pfs,p_box = sess.run([pre_fg_index,pre_fg_score,pre_box_dx],feed_dict={x:r_img})
            
            dx_box_loc = anchor_ut.get_dx_box(p_box[0])
            #print(dx_box_loc[47])
            #print(pfi)
            best_anchor = anchor_ut.get_best_anchor(pfi,pfs,dx_box_loc,100)
            #print(best_anchor[:,5])
            try:
                roi_anchor_index = anchor_ut.py_cpu_nms(best_anchor)
                best_anchor = np.array(best_anchor,dtype=np.int)
                #print(roi_anchor_index)
                #print(best_anchor[roi_anchor_index,5])
                draw_roi(img,best_anchor[roi_anchor_index,5],dx_box_loc)
            except:
                print('no box')
                
        
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

def draw_roi(img,roi_index,dx_box_loc):
    for p_index in roi_index:
        x1,y1,x2,y2 = dx_box_loc[p_index,:]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)), (0, 255, 0), 2)
    show_img(img)

def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

train()