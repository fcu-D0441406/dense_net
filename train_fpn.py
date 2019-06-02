import tensorflow as tf
import os
import cv2
import numpy as np
import dense_net
import at_resnext
import matplotlib.pyplot as plt
import code87_net
import dense_fcn

img_size = 256
class_num=2
channel = 3
batch_size = 2
test_size = 1

ckpts = './checkpoint0_dir'
adam_meta = './checkpoint0_dir/MyModel'


def show_img(img):
    if(img.shape[-1]==1):
        plt.imshow(img,cmap ='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()

def show_feature(sess,x,img,feature):
    #print(img.shape)
    feature_map = sess.run(feature,feed_dict={x:img})
    show_img(feature_map[:,:,0])

def add_neg_weight(mask):
    mask = np.reshape(mask,(-1,img_size*img_size,class_num))
    neg_weight = np.ones(shape=[mask.shape[0],mask.shape[1]],dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(mask[i][j][1]==1):
                neg_weight[i][j]*=10.0
    neg_weight = np.reshape(neg_weight,(-1,img_size,img_size))
    return neg_weight

def show_train_mask(result,batch_x,batch_y):
    result2 = np.reshape(result,(-1,img_size,img_size,class_num))
    image = batch_x[0]
    mask = batch_y[0]
    show_img(image)
    show_img(mask[:,:,1])
    show_img(result2[:,:,1])

def show_test_mask(result,batch_x):
    result2 = np.reshape(result,(-1,img_size,img_size,class_num))
    mask = np.zeros(shape=[result2.shape[0],img_size,img_size],dtype=np.float32)
    for s in range(test_size):
        for i in range(img_size):
            for j in range(img_size):
                if(result2[s][i][j][1]>0.5):
                    mask[s][i][j] = result2[s][i][j][1]
    image = batch_x[0]
    mask = mask[0]
    show_img(image)
    show_img(mask)

def get_data(path):
    image_list = []
    label_list = []
    for root,dirs,files in os.walk(path):
        for f in files:
            if(f=='label.png'):
                label_list.append(os.path.join(root,f))
            elif(f=='img.png'):
                image_list.append(os.path.join(root,f))
    return np.array(image_list),np.array(label_list)

def get_test_data(path):
    image_list = []
    for root,dirs,files in os.walk(path):
        for f in files:
            if(f.endswith('jpg')):
                image_list.append(os.path.join(root,f))
    return np.array(image_list)

def get_test_image(image_list,shuffle_list):
    img = []
    for i in range(batch_size):
        img.append(cv2.imread(image_list[shuffle_list[i]]))
    return np.array(img)
    
    
def convert_label(label):
    label = cv2.imread(label)
    label_r = np.zeros([img_size,img_size,class_num])
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if(label[i][j][2]!=0):
                label_r[i][j][1] = 1
            else:
                label_r[i][j][0] = 1
    return label_r

def get_train_data(image_list,label_list,shuffle_list):
    img = []
    label = []
    for i in range(batch_size):
        img.append(cv2.imread(image_list[shuffle_list[i]]))
        label.append(convert_label(label_list[shuffle_list[i]]))
    return np.array(img),np.array(label)
    
if(__name__=='__main__'):
    image_list,label_list = get_data('./fpn_img')
    image_test_list = get_test_data('./test')
    
    shuffle_test_list = np.arange(0,image_test_list.shape[0])
    shuffle_list = np.arange(0,image_list.shape[0])
    #batch_x,batch_y = get_train_data(image_list,label_list,shuffle_list[j*batch_size:(j+1)*batch_size])

    trainable = tf.placeholder(tf.bool,name='trainable')
    y = tf.placeholder(tf.float32,[None,img_size,img_size,class_num])
    neg_weight = tf.placeholder(tf.float32,[None,img_size,img_size])
    #resnet32 = resnet_v2.ResNet(class_num)
    #resnet32 = at_resnext.Resnext(img_size,channel,class_num)
    #resnet32 = code87_net.Resnext(img_size,channel,class_num)
    resnet32 = dense_fcn.Dense_net(img_size,channel,class_num,batch_size=batch_size,trainable=trainable)
    #resnet32 = dense_net.Dense_net(img_size,channel,class_num,24,0.5)
    x = resnet32.x
    fpn_predict = resnet32.code_87_pre
    #resnet32.upsample(True)
    #resnet32.unsample2()
    
    
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 3), logits=fpn_predict)
    loss = tf.multiply(loss,neg_weight)
    loss = tf.reduce_mean(loss)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,3),tf.argmax(fpn_predict,3)),'float'))
    
    ########## batch_nor 方法
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-1, global_step=global_step, decay_steps=30000,
                                               decay_rate=0.1, staircase=True)
    with tf.name_scope('opt'):
        opt = tf.train.AdamOptimizer(5e-4,name='optimizer')
        #opt = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(loss)
            train_op = opt.apply_gradients(grads, global_step=global_step)
        #tf.summary.scalar('opti', opt)
    
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
        
        print('start --------')
        for i in range(2001):
            np.random.shuffle(shuffle_list)
            np.random.shuffle(shuffle_test_list)
            for j in range(image_list.shape[0]//batch_size):
                batch_x,batch_y = get_train_data(image_list,label_list,shuffle_list[j*batch_size:(j+1)*batch_size])
                neg_w = add_neg_weight(batch_y)
                _,result1,ls,acc = sess.run([train_op,fpn_predict,loss,accuracy],
                                            feed_dict={x:batch_x,y:batch_y,neg_weight:neg_w,trainable:True})
               
                if(j%10==0):
                    batch_test_x = get_test_image(image_test_list,shuffle_list[j*batch_size:(j+1)*batch_size])
                    train_result = sess.run([resnet32.code87_softmax],
                                                       feed_dict={x:batch_x,trainable:False})
                    test_result = sess.run([resnet32.code87_softmax],
                                                       feed_dict={x:batch_test_x,trainable:False})
                    show_test_mask(train_result,batch_x)
                    show_test_mask(test_result,batch_test_x)
                    #print(resnext1.shape)
                        
                    print('train ',ls,acc)
                    saver.save(sess,adam_meta,global_step=i) 
                    print('---------')
    