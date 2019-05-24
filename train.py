import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
import matplotlib.pyplot as plt

import resnext_cbam
import resnext_cbam_v2
import resnext_cbam_sprgb
import densenet_cbam
import densenet_dim
import densenet_dim_v2
#ck1 rensext_cbam
#ck2 resnext_cbam_v2
'''
overkill=4.96% underkill=60ppm
ck4 MyModel-3 defect_rate=0.0132 densenet_dim
'''
ckpts = './checkpoint4_dir'
adam_meta = './checkpoint4dir/MyModel'
batch_size = 32
lr = 2e-5
CLASS_NUM = 2
epoch = 150
img_size = 256
weight_decay = 0.0001
channel=3

def get_data(file_path):
    image = []
    temp = []
    for root,dirs,files in os.walk(file_path):
        for name in files:
            image.append(os.path.join(root,name))

        for sub_dir in dirs:
            temp.append(os.path.join(root,sub_dir))
            
    labels = []
    for folder in temp:
        n_img = len(os.listdir(folder))
        letter = folder.split('/')[-1]
        labels = np.append(labels,n_img*[letter])
    print(len(labels),len(image))
    temp = np.array([image,labels])
    temp = temp.transpose()#維樹反轉 [2,25000] -> [25000,2]
    np.random.shuffle(temp) #numpy的打亂函數
    print(temp.shape)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    
    return np.array(image_list),np.array(label_list)

def get_train_data(img_list,label_list,shuffle_list,j):
    img = []
    label = []
    for i in range(j*batch_size,(j+1)*batch_size):
        img.append(cv2.imread(img_list[shuffle_list[i]]))
        label.append(label_list[shuffle_list[i]])
    
    
    return np.array(img),np.array(label)


def label_convert_onehot(labels):
    n_sample = labels.shape[0]
    n_class = CLASS_NUM
    one_hot_labels = np.zeros((n_sample,n_class))
    one_hot_labels[np.arange(n_sample),labels] = 1
    return np.array(one_hot_labels,dtype = np.uint8)

    
def get_weight(y):
    n_weight = np.ones([batch_size])
    for k in range(y.shape[0]):
        if(y[k][1]==1):
            n_weight[k] = n_weight[k]*2.5
    return n_weight


def check_enviroment():
    if(not os.path.exists(ckpts)):
        os.mkdir(ckpts)
        print('create dir')
    else:
        print('already exists')

def show_img(img):
    #plt.savefig("filename.png")
    if(img.shape[-1]==1):
        plt.imshow(img,cmap ='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()


def get_TTA_data(img_list,label_list,shuffle_list,j,k):
    img = []
    label = []
    for i in range(j*batch_size,(j+1)*batch_size):
        image = cv2.imread(img_list[shuffle_list[i]])
        image = np.rot90(image,k+1)
        img.append(image)
        label.append(label_list[shuffle_list[i]])
    
    return np.array(img),np.array(label)
        

def test_function(sess,test_image_list,test_label_list,shuffle_test_list,loss,x,y,neg_weight,defect_rate,trainable,test):
    overkill = 0
    underkill = 0
    test_loss = 0
    if(test==True):
        f = open('defect.txt','w')
        f2 = open('pass.txt','w')
        pass_num = 0
        defect_num = 0
    for j in range(test_image_list.shape[0]//batch_size):
        batch_x,batch_y = get_train_data(test_image_list,test_label_list,shuffle_test_list,j)
        batch_x = (np.array(batch_x,dtype=np.float32)/255.0)
        batch_y = label_convert_onehot(batch_y)
        train_n_weight = get_weight(batch_y)
        '''
        ls,prediction = sess.run([loss,resnet32.pre_softmax],
                                     feed_dict={x:batch_x,y:batch_y,neg_weight:train_n_weight,
                                                trainable:False})
        '''
        ls,prediction,de_img = sess.run([loss,resnet32.pre_softmax,resnet32.decoder_img],
                                     feed_dict={x:batch_x,y:batch_y,neg_weight:train_n_weight,
                                                trainable:False})
        '''
        for k in range(3):
            batch_x,batch_y = get_TTA_data(test_image_list,test_label_list,shuffle_test_list,j,k)
            batch_x = (np.array(batch_x,dtype=np.float32)/255.0)
            batch_y = label_convert_onehot(batch_y)
            train_n_weight = get_weight(batch_y)
            
            ls,prediction2,de_img = sess.run([loss,resnet32.pre_softmax,resnet32.decoder_img],
                                         feed_dict={x:batch_x,y:batch_y,neg_weight:train_n_weight,
                                                    trainable:False})
            prediction+=prediction2
        prediction = prediction/4.0
        '''

        for k in range(prediction.shape[0]):
            if(test==True):
                if(np.argmax(batch_y[k]==0)):
                    p = str(pass_num)+','+str(prediction[k][0])+'\n'
                    f.writelines(p)
                    pass_num+=1
                else:
                    p = str(defect_num)+','+str(prediction[k][1])+'\n'
                    f2.writelines(p)
                    defect_num+=1
            if(np.argmax(batch_y[k])==0):
                if(prediction[k][0]<=(1-defect_rate)):
                    overkill+=1
                    #if(test==True):
                    #    show_img()
            else:
                if(prediction[k][1]<=defect_rate):
                    underkill+=1
                    if(test==True):
                        #save_path = './result/underkill/'
                        #cv2.imwrite(save_path+str(underkill)+'.jpg',batch_x[k])
                        show_img(batch_x[k])
                        print(prediction[k])

            #print(batch_y[k],prediction[k])
        test_loss+=ls
    show_img(de_img[0])
    if(test==True):
        print('overkill',overkill,'underkill',underkill)
    return overkill,underkill,test_loss
if(__name__=='__main__'):
    check_enviroment()
    with tf.Graph().as_default():
        trainable = tf.placeholder(tf.bool,name='trainable')
        test = True
        defect_rate = 0.0132
        #resnet32 = resnext_cbam.Resnext(img_size,channel,2,batch_size,trainable=trainable)
        resnet32 = densenet_dim.Dense_net(img_size,channel,2,batch_size=batch_size,trainable=trainable)
        #resnet32 = densenet_dim_v2.Dense_net(img_size,channel,2,batch_size=batch_size,trainable=trainable)
        '''
        if(test==True):
            #resnet32 = resnext_cbam.Resnext(img_size,channel,2,batch_size,True)
            #resnet32 = resnext_cbam_v2.Resnext(img_size,channel,2,batch_size,False)
            #resnet32 = densenet_cbam.Dense_net(img_size,channel,2,trainable=False)
            defect_rate = 0.37
        else:
            resnet32 = resnext_cbam.Resnext(img_size,channel,2,batch_size,True) #ck2
            #resnet32 = densenet_cbam.Dense_net(img_size,channel,2)
            #resnet32 = resnext_cbam_v2.Resnext(img_size,channel,2,batch_size,True)
            #resnet32 = resnext_cbam_sprgb.Resnext(img_size,channel,2,batch_size,True)
            defect_rate = 0.5
        '''
        image_list,label_list = get_data('./train')
        shuffle_list = np.arange(image_list.shape[0])
        test_image_list,test_label_list = get_data('./TU117_test')
        shuffle_test_list = np.arange(test_image_list.shape[0])
        x = resnet32.x
        y = tf.placeholder(tf.float32,[None,CLASS_NUM])

        neg_weight = tf.placeholder(tf.float32,[None])

        with tf.name_scope('loss'):
            loss2 = tf.reduce_mean(tf.image.ssim(resnet32.decoder_img,x, max_val=1.0))
            loss2 = tf.clip_by_value(loss2,0.0,1.0)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), 
                                                                                 logits=resnet32.prediction)
            loss = tf.multiply(loss,neg_weight)
            loss = tf.reduce_mean(loss)
            loss = loss+loss2
            #l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            #loss = loss+l2_loss*weight_decay
            tf.summary.scalar('loss', loss)

        with tf.name_scope('acc'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(resnet32.prediction,1)),'float'))
            tf.summary.scalar('ACC', accuracy)
            
        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=5e-3, global_step=global_step, decay_steps=25000,decay_rate=0.1, staircase=False)
        #opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')
        with tf.name_scope('opt'):
            #opt = tf.train.MomentumOptimizer(learning_rate,0.9,name='optimizer')
            #opt = tf.train.AdamOptimizer(lr,name='optimizer')
            #opt = tf.contrib.opt.MomentumWOptimizer(1e-4,learning_rate=learning_rate,momentum=0.9,name='optimizer')
            opt =tf.contrib.opt.AdamWOptimizer(1e-4,learning_rate=5e-4,name='optimizer')
            ########## batch_nor 方法
            if(test==False):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads = opt.compute_gradients(loss)
                    train_op = opt.apply_gradients(grads, global_step=global_step)
            else:
                pass
            #tf.summary.scalar('opti', opt)
            ##########
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter('./log', sess.graph)
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            
            ######## batch_norm save
            var_list = tf.trainable_variables()
            if global_step is not None:
                var_list.append(global_step)
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list=var_list,max_to_keep=100)
            ########

            if tf.train.latest_checkpoint(ckpts) is not None:
                #ck1 28105 defect_rate = 0.0005
                #ck3 latest_ck
                #ck4 3 defect_rate=0.0132
                #saver.restore(sess, './checkpoint2_dir/MyModel-10682')
                #saver.restore(sess, './checkpoint4_dir/MyModel-3')
                saver.restore(sess, tf.train.latest_checkpoint(ckpts))
            else:
                assert 'can not find checkpoint folder path!'
                
            
            min_underkill = 3
            max_test_acc = 0.9129
            min_test_loss = 0.05
            min_error = 900
            for i in range(epoch):
                np.random.shuffle(shuffle_list)
                np.random.shuffle(shuffle_test_list)    
                if(test==False):
                    for j in range(image_list.shape[0]//batch_size):
                        batch_x,batch_y = get_train_data(image_list,label_list,shuffle_list,j)
                        batch_x = (np.array(batch_x,dtype=np.float16)/255.0)
                        batch_y = label_convert_onehot(batch_y)
                        train_n_weight = get_weight(batch_y)
                        _,g_step,ls,acc,lr_rate = sess.run([train_op,global_step,loss,accuracy,learning_rate],
                                            feed_dict={x:batch_x,y:batch_y,neg_weight:train_n_weight,
                                                       trainable:True})

                        if(j%50==0):
                            print('------')
                            print('epoch ',i,' ls ',ls,' acc ',acc)
                        
                        if((j%200==0 or j==(image_list.shape[0]//batch_size)-1)):
                            pass
                        
                overkill,underkill,test_loss = test_function(sess,test_image_list,test_label_list,shuffle_test_list,
                                                     loss,x,y,neg_weight,defect_rate,trainable,test)
                            
                test_loss = test_loss/(test_image_list.shape[0]//batch_size)
                test_acc = float(test_image_list.shape[0]-overkill-underkill)/test_image_list.shape[0]
    
                error = overkill+(30*underkill)
                
                if(test==True):
                    break
                if(g_step>=10000):
                    if(test_acc>max_test_acc and min_error>error and min_test_loss>test_loss):
                        max_test_acc = test_acc
                        min_underkill = underkill
                        min_error = error
                        saver.save(sess,adam_meta,global_step=0)
                    else:
                        if(test_acc>max_test_acc):
                            max_test_acc = test_acc
                            saver.save(sess,adam_meta,global_step=1) 
                        if(min_underkill>underkill):
                            min_underkill = underkill
                            saver.save(sess,adam_meta,global_step=2) 
                        if(min_test_loss>test_loss):
                            min_test_loss=test_loss
                            saver.save(sess,adam_meta,global_step=3) 
                        if(error<min_error):
                            min_error = error
                            saver.save(sess,adam_meta,global_step=4) 
                        
                print('epoch',i,'g_step',g_step,'lr',lr_rate,'test_loss',test_loss,'test_acc',test_acc,
                          'underkill',underkill,'overkill',overkill)
                print('min_test_loss',min_test_loss,'max_test_acc',max_test_acc,'min_underkill',min_underkill,'min_error',min_error)
        
                if(i%5==0 and i!=0):
                    saver.save(sess,adam_meta,global_step=(i+10))
                