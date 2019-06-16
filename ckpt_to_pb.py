import tensorflow as tf
import os
import cv2
import numpy as np
import densenet_dim

#from create_tf_record import *
from tensorflow.python.framework import graph_util

input_tensor_name = 'input:0'
output_tensor_name = 'Softmax:0'
img_path = './29.jpg'

def freeze_graph(pre_soft,x):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    output_graph = './123.pb'
    input_checkpoint = './checkpoint3_dir/MyModel-101'
    #checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    
    output_node_names = "Softmax"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        
        im=cv2.imread(img_path)
        im=im[np.newaxis,:]
        predict = sess.run(pre_soft,feed_dict={x:im})
        print(predict)
        
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
 
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
 
            # 定义输出的张量名称
            output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)
 
            # 读取测试图片
            im=cv2.imread(img_path)
            im=im[np.newaxis,:]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out=sess.run(output_tensor, feed_dict={input_tensor: im})
            print(out)


if(__name__=='__main__'):
    with tf.Graph().as_default():
        d = densenet_dim.Dense_net(32,3,10,16,0.5,False,None)
        pre_soft = tf.nn.softmax(d.prediction)
        print(pre_soft)
        freeze_graph(pre_soft,d.x)
        freeze_graph_test('./123.pb',img_path)
        
        
        