# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tqdm
import sys
import res2net_DFL_oct as res2net
import lookahead

training_data_path = sys.argv[1]
try:
    training_mark = '_' + sys.argv[2]
except:
    training_mark = ''

device_name = training_data_path.split('/')[-1] + training_mark

master_path = './'
ckpts = master_path + '{}/checkpoint0_dir_DFL10'.format(device_name)
adam_meta = master_path + '{}/checkpoint0_dir_DFL10/MyModel'.format(device_name)
batch_size = 32
drop_rate = 0.3
use_mix_up = True
warmup_num = 1000
epoch = 0
batch_number = 25000 #40000
batch_number2 = 5000 #40000
lr = 7e-4
img_size = 600
weight_decay = 1e-4
channel=3

def get_data_dir(path):
    training_path,val_path,test_path = '','',''
    class_num = 0
    for dirs in os.listdir(path):
        if('val_test_combined' in dirs):
            test_path = os.path.join(path,dirs)
        if('val_test_combined' in dirs):
            val_path = os.path.join(path,dirs)
        if('multi_defect_train' in dirs):
            if(not 'EtE' in dirs):
                training_path = os.path.join(path,dirs)
                
    for dirs in os.listdir(training_path):
        if(ord(dirs[0])>=48 and ord(dirs[0])<=57):
            class_num+=1
    return training_path,val_path,test_path,class_num

def check_enviroment():
    need_dir = [master_path + '{}'.format(device_name),ckpts,master_path + '{}/result'.format(device_name),master_path + '{}/result'.format(device_name),
               master_path + '{}/result/overkill'.format(device_name),master_path + '{}/result/underkill'.format(device_name)]
    for nd in need_dir:
        if(not os.path.exists(nd)):
            os.mkdir(nd)
            print('create dir')

def load_data(path,class_num,limit=False):
    label = []
    image = []
    img_path = []
    defect = []
    defect_label = []
    now_index = 0
    for root,dirs,files in os.walk(path):
        for d in dirs:
            for root_,_,files_ in os.walk(os.path.join(root,d)):
                for f in files_:
                    try:
                        label_num = int(d)
                        if(f.endswith('.jpg') or f.endswith('.JPG')):
                            img = mpimg.imread(os.path.join(root_,f))
                            if(img.shape[0]==600 and img.shape[1]==600):
                                label_ = np.zeros((class_num))
                                label_[label_num] = 1

                                if(label_num!=class_num-1):
                                    defect_label.append(label_)
                                    defect.append(img)

                                label.append(label_)
                                image.append(img)
                                img_path.append(os.path.join(root_,f))
                    except:
                        continue
            now_index+=1
    return np.array(image),np.array(label),np.array(img_path),np.array(defect),np.array(defect_label)

training_path,val_path,test_path,CLASS_NUM = get_data_dir(training_data_path)

print(training_path,val_path,test_path,CLASS_NUM)
check_enviroment()
assert(training_path!='' or val_path!='' or test_path!='' or CLASS_NUM!=0)

print('start load image')
image_list,label_list,train_img_path,train_defect,train_defect_label = load_data(training_path,CLASS_NUM)
test_image_list,test_label_list,test_img_path,_,_ = load_data(val_path,CLASS_NUM)
print(image_list.shape,label_list.shape,train_defect.shape,train_defect_label.shape)
print(test_image_list.shape,test_label_list.shape)
epoch = ((batch_number*batch_size)//image_list.shape[0])
epoch2 = ((batch_number2*batch_size)//image_list.shape[0])
trainable = tf.placeholder(tf.bool,name='trainable')
#resnet32 = densenet.Dense_net(img_size,channel,CLASS_NUM,batch_size,drop_rate=drop_rate,trainable=trainable)
resnet32 = res2net.Res2net(img_size,channel,batch_size,CLASS_NUM,trainable=trainable)

x = resnet32.input
y = tf.placeholder(tf.float32,[None,CLASS_NUM])
y_side = tf.placeholder(tf.float32,[None,CLASS_NUM])
mix_weight = tf.placeholder(tf.float32,[None])
y_ = tf.placeholder(tf.float32,[None,2])
neg_weight = tf.placeholder(tf.float32,[None])


def huber_weight(y):
    return tf.map_fn(lambda x:tf.cond(tf.greater(1-x,0.05),lambda:1-y,lambda:tf.pow(1-x,2)),y)

def multi_category_focal_loss1(y_true,y_pred,gamma=2.0):
    alpha = []
    for i in range(CLASS_NUM):
        if(i==CLASS_NUM-1):
            alpha.append([0.5])
        else:
            alpha.append([1])
    epsilon = 1e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = float(gamma)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.where(tf.greater(1-y_t,0.01),1-y_t,tf.pow(1-y_t,2))
    loss = tf.matmul(tf.multiply(weight, ce), alpha)
    return loss


def mixup_focal_loss1(y_true_1,y_true_2,y_pred,mix_weight,gamma=2.0):

    loss1 = multi_category_focal_loss1(y_true_1,y_pred,gamma=gamma)
    loss2 = multi_category_focal_loss1(y_true_2,y_pred,gamma=gamma)
    loss = tf.multiply(loss1,mix_weight) + tf.multiply(loss2,1-mix_weight)
    
    return loss


def two_category_focal_loss1(y_true,y_pred,gamma=2.0):
    alpha = [[1],[0.5]]

    epsilon = 1e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = float(gamma)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.where(tf.greater(1-y_t,0.01),1-y_t,tf.pow(1-y_t,2))
    loss = tf.matmul(tf.multiply(weight, ce), alpha)
    return loss

with tf.name_scope('loss'):

    loss1 = mixup_focal_loss1(y,y_side, resnet32.pre_softmax_,mix_weight)
    loss_DFL = mixup_focal_loss1(y,y_side, resnet32.pre_softmax_DFL,mix_weight)
    loss_GMP = mixup_focal_loss1(y,y_side, resnet32.pre_softmax_GMP,mix_weight)
    
    loss = loss1+0.5*loss_DFL+0.5*loss_GMP
    loss = tf.reduce_mean(loss)
    
    two_cls_loss = two_category_focal_loss1(y_, resnet32.two_cls_pre_softmax)
    two_cls_loss = tf.reduce_mean(two_cls_loss)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('two_cls_loss', two_cls_loss)

with tf.name_scope('acc'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(resnet32.pre_softmax,1)),'float'))
    two_cls_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1),tf.argmax(resnet32.two_cls_pre_softmax,1)),'float'))

    tf.summary.scalar('ACC', accuracy)
    ########## batch_nor 方法
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ##########
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                              initializer=tf.constant_initializer(0), trainable=False)
    epoch_num = image_list.shape[0]//batch_size
    d_step = epoch_num*250
    warmup_step = warmup_num
    normal_lr = tf.train.exponential_decay(learning_rate=lr, global_step=global_step,
                                           decay_steps=batch_number,decay_rate=0.5, staircase=True)
    warmup_lr = normal_lr*(tf.cast(global_step, tf.float32)/tf.cast(warmup_step, tf.float32))
    learning_rate = tf.cond(global_step < warmup_step, lambda: warmup_lr, lambda: normal_lr)


with tf.name_scope('opt'):
    train_multi_var_list = [v for v in tf.trainable_variables() if 'res2net' in v.name ]
    train_two_var_list = [v for v in tf.trainable_variables() if 'cls_layer' in v.name]
    #opt = tf.train.MomentumOptimizer(learning_rate,0.9,name='optimizer')
    opt = tf.contrib.opt.AdamWOptimizer(weight_decay,learning_rate,name='optimizer')
    opt = lookahead.LookaheadOptimizer(opt)
    with tf.control_dependencies(update_ops):
        multi_grads = opt.compute_gradients(loss,var_list=train_multi_var_list)
        two_grads = opt.compute_gradients(two_cls_loss,var_list=train_two_var_list)
        train_op = opt.apply_gradients(multi_grads, global_step=global_step)
        train_op2 = opt.apply_gradients(two_grads, global_step=global_step)


######## batch_norm save
var_list = tf.trainable_variables()
if global_step is not None:
    var_list.append(global_step)
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
blur_vars = [g for g in g_list if 'anti_filter' in g.name]
var_list += blur_vars
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list,max_to_keep=100)
########

def get_train_data(img_list,label_list,shuffle_list,j):
    k = 0
    img = np.zeros([batch_size,img_size,img_size,channel],dtype=np.int32)
    label = []
    two_cls_label = []
    for i in range(j*batch_size,(j+1)*batch_size):
        img[k] = img_list[shuffle_list[i]]
        label.append(label_list[shuffle_list[i]])
        if(label_list[shuffle_list[i]][CLASS_NUM-1]==1):
            two_cls_label.append([0,1])
        else:
            two_cls_label.append([1,0])
        k+=1

    return np.array(img),np.array(label),np.array(two_cls_label)

def random_brightness(x,bright_range=(0.9,1.1)):
    result = []
    for i in range(x.shape[0]):
        hsv = cv2.cvtColor(x[i].astype(np.uint8),cv2.COLOR_RGB2HSV)
        v_channel = hsv[:,:,2].astype(np.float32) * np.random.uniform(bright_range[0], bright_range[1])
        v_channel = [np.where(v_ch>255.,255.,v_ch) for v_ch in v_channel]
        hsv[:,:,2] = v_channel
        result.append(cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB))
    return np.array(result)

def mixup(x,y,alpha=1.0):
    x = np.array(x,dtype=np.float32)
    y = np.array(y,dtype=np.float32)
    x_ = x.copy()
    y_side = y.copy()
    pass_index = []
    defect_index = []
    mix_weight = []
    for i in range(x.shape[0]):
        if(y[i][CLASS_NUM-1]==1):
            pass_index.append(i)
            mix_weight.append(1.0)
        else:
            defect_index.append(i)
            mix_weight.append(1.0)
    for i in range(x.shape[0]):
        if(np.random.randint(10)<5):
            if(y[i][CLASS_NUM-1]==1):
                random_idx = np.random.randint(len(pass_index))
                random_rate = np.random.beta(alpha,alpha)
                random_rate = max(random_rate,1-random_rate)
                mix_img = x[i]*random_rate+x[pass_index[random_idx]]*(1-random_rate)
#                 mix_label = y[i]*random_rate+y[pass_index[random_idx]]*(1-random_rate)
                x_[i] = mix_img
                y_side[i] = y[pass_index[random_idx]]
                mix_weight[i] = random_rate
            else:
                random_idx = np.random.randint(len(defect_index))
                random_rate = np.random.beta(alpha,alpha)
                random_rate = max(random_rate,1-random_rate)
                mix_img = x[i]*random_rate+x[defect_index[random_idx]]*(1-random_rate)
#                 mix_label = y[i]*random_rate+y[defect_index[random_idx]]*(1-random_rate)
                x_[i] = mix_img
                y_side[i] = y[defect_index[random_idx]]
                mix_weight[i] = random_rate
    return x_, y_side, mix_weight


def get_train_with_aug(img_list,label_list,shuffle_list,j):
    k = 0
    img = np.zeros([batch_size,img_size,img_size,channel],dtype=np.int32)
    label = []
    two_cls_label = []
    for i in range(j*batch_size,(j+1)*batch_size):
        if(np.random.randint(batch_size)<batch_size//2):
            random_index = np.random.randint(train_defect.shape[0])
            img[k] =  train_defect[random_index].copy()
            label.append(train_defect_label[random_index])
            two_cls_label.append([1,0])
        else:
            img[k] = img_list[shuffle_list[i]].copy()
            label.append(label_list[shuffle_list[i]])
            if(label_list[shuffle_list[i]][CLASS_NUM-1]==1):
                two_cls_label.append([0,1])
            else:
                two_cls_label.append([1,0])
            
        random_k = np.random.randint(4)
        random_f = np.random.randint(3)
        if(random_f!=2):
            img[k] = np.flip(img[k],random_f)
        img[k] = np.rot90(img[k],random_k)
        k+=1
    img = random_brightness(np.array(img))
    img = np.array(img,dtype=np.float32)
    label = np.array(label,dtype=np.float32)
    if use_mix_up:
        img,label_side,mix_weight = mixup(img,label)
    return img,label,label_side,mix_weight,np.array(two_cls_label)

def get_weight(y):
    n_weight = np.ones([batch_size])
    for k in range(y.shape[0]):
        if(y[k][CLASS_NUM-1]!=1):
            n_weight[k] = n_weight[k]*6.0
    return n_weight

def show_img(img):
    #plt.savefig("filename.png")
    if(img.shape[-1]==1):
        plt.imshow(img,cmap ='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()

def test_function(sess,test_image_list,test_label_list,shuffle_test_list,loss,accuracy,
                  x,y,neg_weight,defect_rate,trainable,test,img_path=''):
    overkill = 0
    underkill = 0
    test_loss = 0
    total = 0
    mean_acc = 0
    
    for j in range(test_image_list.shape[0]//batch_size):
        batch_x,batch_y,batch_two_cls_y = get_train_data(test_image_list,test_label_list,shuffle_test_list,j)
        train_n_weight = get_weight(batch_y)
        
        ls,prediction,test_acc,inputs_ = sess.run([loss,resnet32.two_cls_pre_softmax,accuracy,resnet32.inputs],
                                         feed_dict={x:batch_x,y:batch_y,neg_weight:train_n_weight,y_:batch_two_cls_y,
                                                trainable:False})
        mean_acc+=test_acc
        test_loss+=ls
        for k in range(prediction.shape[0]):
            if(np.argmax(batch_y[k])==CLASS_NUM-1):
                if(prediction[k][1]<=(1-defect_rate)):
                    overkill+=1
            else:
                if(prediction[k][1]>=(1-defect_rate)):
                    underkill+=1
            total+=1
        
    mean_acc = mean_acc/(test_image_list.shape[0]//batch_size)
    test_loss = test_loss/(test_image_list.shape[0]//batch_size)
    print('test_loss',test_loss,'test_acc',mean_acc)
    return overkill,underkill,test_loss,mean_acc

check_enviroment()

defect_rate = 0.5

shuffle_list = np.arange(image_list.shape[0])

shuffle_test_list = np.arange(test_image_list.shape[0])

min_underkill = 0
max_test_acc = 0
min_test_loss = 1.0
min_error30 = 10000
min_error60 = 10000
min_error90 = 10000
min_error150 = 10000

test = False
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,master_path+'InitialModel/checkpoint0_dir_DFL10/MyModel-35')
    initial_global_step = global_step.assign(0)
    sess.run(initial_global_step)
    
    print('start train')
    count = -1
    for i in range(int(epoch)):
        np.random.shuffle(shuffle_list)
        np.random.shuffle(shuffle_test_list)
        if(test==False):
            for j in range(image_list.shape[0]//batch_size):
                count = count + 1
                batch_x,batch_y,batch_y_side,batch_mix_weight,batch_two_cls_y = get_train_with_aug(image_list,label_list,shuffle_list,j)
                train_n_weight = get_weight(batch_y)
                
                
                _,g_step,ls,acc,lr_rate = sess.run([train_op,global_step,loss,accuracy,learning_rate],
                                                feed_dict={x:batch_x,y:batch_y,y_side:batch_y_side,mix_weight:batch_mix_weight,
                                                           neg_weight:train_n_weight,y_:batch_two_cls_y,trainable:True})
                
                _,two_cls_ls,two_cls_acc = sess.run([train_op2,two_cls_loss,two_cls_accuracy],
                                                feed_dict={x:batch_x,y:batch_y,y_side:batch_y_side,mix_weight:batch_mix_weight,
                                                           neg_weight:train_n_weight,y_:batch_two_cls_y,trainable:True})
                
                if(count%50==0):
                    print('------')
                    print('epoch ',i,'/',int(epoch),' ls ',ls,' acc ',acc,'two_cls_ls',two_cls_ls,'two_cls_acc',two_cls_acc,
                         ' learning rate ',lr_rate)

                if(count>=3000 and count%400==0):                 
                    overkill,underkill,test_loss,test_acc = test_function(sess,test_image_list,test_label_list,shuffle_test_list,
                                                                          two_cls_loss,two_cls_accuracy,x,y,neg_weight,
                                                                          defect_rate,trainable,test)

                    over_under_acc = float(test_image_list.shape[0]-overkill-underkill)/test_image_list.shape[0]

                    error30 = (overkill+(30.*underkill))/test_image_list.shape[0]*100
                    error60 = (overkill+(60.*underkill))/test_image_list.shape[0]*100
                    error90 = (overkill+(90.*underkill))/test_image_list.shape[0]*100
                    error150 = (overkill+(150.*underkill))/test_image_list.shape[0]*100
                    if(g_step>=1500):
                        if(min_error30>error30):
                            min_error30=error30
                            saver.save(sess,adam_meta,global_step=1) 
                        if(min_error60>error60):
                            min_error60=error60
                            saver.save(sess,adam_meta,global_step=2)
                        if(min_error90>error90):
                            min_error90=error90
                            saver.save(sess,adam_meta,global_step=3)
                        if(min_error150>error150):
                            min_error150=error150
                            saver.save(sess,adam_meta,global_step=4)

                    print('epoch',i,'g_step',g_step,'lr',lr_rate,'test_loss',test_loss,'test_acc',test_acc,
                          'over_under_acc',over_under_acc,'underkill',underkill,'overkill',overkill)
                    print('min_test_loss',min_test_loss,'max_test_acc',max_test_acc,'min_underkill',min_underkill,
                          'min_error30',min_error30,'min_error60',min_error60,'min_error90',min_error90,'min_error150',min_error150)

with tf.Session(config=config) as sess:
    
    sess.run(tf.global_variables_initializer())
    if tf.train.latest_checkpoint(ckpts) is not None:
        #saver.restore(sess, './checkpoint4_dir/MyModel-4')
        saver.restore(sess, master_path + '{}/checkpoint0_dir_DFL10/MyModel-1'.format(device_name))
        #saver.restore(sess, tf.train.latest_checkpoint(ckpts))
    else:
        assert 'can not find checkpoint folder path!'
    print('start train')
    count = -1
    for i in range(int(epoch2)):
        np.random.shuffle(shuffle_list)
        np.random.shuffle(shuffle_test_list)
        if(test==False):
            for j in range(image_list.shape[0]//batch_size):
                count = count + 1
                batch_x,batch_y,batch_y_side,batch_mix_weight,batch_two_cls_y = get_train_with_aug(image_list,label_list,shuffle_list,j)
                train_n_weight = get_weight(batch_y)
                
                
                _,g_step,ls,acc,lr_rate = sess.run([train_op,global_step,loss,accuracy,learning_rate],
                                                feed_dict={x:batch_x,y:batch_y,y_side:batch_y_side,mix_weight:batch_mix_weight,
                                                           neg_weight:train_n_weight,y_:batch_two_cls_y,trainable:True})
                
                _,two_cls_ls,two_cls_acc = sess.run([train_op2,two_cls_loss,two_cls_accuracy],
                                                feed_dict={x:batch_x,y:batch_y,y_side:batch_y_side,mix_weight:batch_mix_weight,
                                                           neg_weight:train_n_weight,y_:batch_two_cls_y,trainable:True})
                
                if(count%50==0):
                    print('------')
                    print('epoch ',i,'/',int(epoch),' ls ',ls,' acc ',acc,'two_cls_ls',two_cls_ls,'two_cls_acc',two_cls_acc,
                         ' learning rate ',lr_rate)

                if(count%400==0):                 
                    overkill,underkill,test_loss,test_acc = test_function(sess,test_image_list,test_label_list,shuffle_test_list,
                                                                          two_cls_loss,two_cls_accuracy,x,y,neg_weight,
                                                                          defect_rate,trainable,test)

                    over_under_acc = float(test_image_list.shape[0]-overkill-underkill)/test_image_list.shape[0]

                    error30 = (overkill+(30.*underkill))/test_image_list.shape[0]*100
                    error60 = (overkill+(60.*underkill))/test_image_list.shape[0]*100
                    error90 = (overkill+(90.*underkill))/test_image_list.shape[0]*100
                    error150 = (overkill+(150.*underkill))/test_image_list.shape[0]*100
                    if(g_step>=1500):
                        if(min_error30>error30):
                            min_error30=error30
                            saver.save(sess,adam_meta,global_step=1) 
                        if(min_error60>error60):
                            min_error60=error60
                            saver.save(sess,adam_meta,global_step=2)
                        if(min_error90>error90):
                            min_error90=error90
                            saver.save(sess,adam_meta,global_step=3)
                        if(min_error150>error150):
                            min_error150=error150
                            saver.save(sess,adam_meta,global_step=4)

                    print('epoch',i,'g_step',g_step,'lr',lr_rate,'test_loss',test_loss,'test_acc',test_acc,
                          'over_under_acc',over_under_acc,'underkill',underkill,'overkill',overkill)
                    print('min_test_loss',min_test_loss,'max_test_acc',max_test_acc,'min_underkill',min_underkill,
                          'min_error30',min_error30,'min_error60',min_error60,'min_error90',min_error90,'min_error150',min_error150)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver.restore(sess, master_path + '{}/checkpoint0_dir_DFL10/MyModel-1'.format(device_name))
test = True#45
fresh_image_list,fresh_label_list,fresh_img_path,_,_ = load_data(test_path,CLASS_NUM)
print(fresh_image_list.shape,fresh_label_list.shape,fresh_img_path.shape)

def get_train_data_(img_list,label_list,shuffle_list,j,img_path):
    k = 0
    img = np.zeros([batch_size,img_size,img_size,channel],dtype=np.int32)
    label = []
    path = []
    two_cls_label = []
    for i in range(j*batch_size,(j+1)*batch_size):
        img[k] = img_list[shuffle_list[i]]
        label.append(label_list[shuffle_list[i]])
        path.append(img_path[shuffle_list[i]])
        if(label_list[shuffle_list[i]][CLASS_NUM-1]==1):
            two_cls_label.append([0,1])
        else:
            two_cls_label.append([1,0])
        k+=1

    return np.array(img),np.array(label),np.array(two_cls_label),np.array(path)

def test_function_fresh(sess,test_image_list,test_label_list,shuffle_test_list,loss,accuracy,
                  x,y,neg_weight,defect_rate,trainable,test):
            overkill = 0
            underkill = 0
            test_loss = 0
            total = 0
            mean_acc = 0

            if(test==True):
                f = open(master_path + '{}/pass.txt'.format(device_name),'w')
                f2 = open(master_path + '{}/defect.txt'.format(device_name),'w')
                pass_num = 0
                defect_num = 0
            for j in range(test_image_list.shape[0]//batch_size):
                batch_x,batch_y,batch_two_cls_y,batch_path = get_train_data_(test_image_list,test_label_list,shuffle_test_list,j,fresh_img_path)
                train_n_weight = get_weight(batch_y)

                ls,prediction,test_acc = sess.run([loss,resnet32.two_cls_pre_softmax,accuracy],
                                                 feed_dict={x:batch_x,y:batch_y,neg_weight:train_n_weight,y_:batch_two_cls_y,
                                                        trainable:False})
                mean_acc+=test_acc
                
                for k in range(prediction.shape[0]):
                    if(np.argmax(batch_y[k])==CLASS_NUM-1):
                        p = str(pass_num)+','+str(1-prediction[k][1])+'\n'
                        f.writelines(p)
                        pass_num+=1
                        if(prediction[k][1]<=(1-defect_rate)):
                            save_img = cv2.imread(batch_path[k])
                            cv2.imwrite(os.path.join(master_path + '{}/result/overkill'.format(device_name),batch_path[k].split('/')[-1]),save_img)
                            overkill+=1
                    else:
                        p = str(defect_num)+','+str(1-prediction[k][1])+'\n'
                        f2.writelines(p)
                        defect_num+=1
                        if(prediction[k][1]>=(1-defect_rate)):
                            save_img = cv2.imread(batch_path[k])
                            cv2.imwrite(os.path.join(master_path + '{}/result/underkill'.format(device_name),batch_path[k].split('/')[-1]),save_img)
                            underkill+=1
                    total+=1
                test_loss+=ls
                if(test==True):
                    print('total',total,'overkill',overkill,'underkill',underkill)
            if(test==True):
                print('overkill',overkill,'underkill',underkill,'overkill rate',float(overkill)/total
                     ,float(underkill)/total)

            mean_acc = mean_acc/(test_image_list.shape[0]//batch_size)
            return overkill,underkill,test_loss,mean_acc

test = True
defect_rate = 0.5
shuffle_fresh_list = np.arange(fresh_image_list.shape[0])

overkill,underkill,test_loss,test_acc = test_function_fresh(sess,fresh_image_list,fresh_label_list,shuffle_fresh_list,
                                                         two_cls_loss,two_cls_accuracy,x,y,neg_weight,defect_rate,trainable,test)
print('overkill',overkill,'underkill',underkill)


def visualize_distribution(defect_score, pass_score):
    
    import numpy as np
    import matplotlib.pyplot as plt
    save_path = master_path + '{}/'.format(device_name)
    defect_score = np.array(defect_score,dtype=np.float32)
    pass_score   = np.array(pass_score,dtype=np.float32)
    
    hist_para = {
        'bins'    : 100,
        'histtype': 'bar',
        'bottom'  : 0,
    #     'density' : True,
        'rwidth'  : 0.8,
        'log'     : True,
#         'alpha'   : 0.7,
    }

    plt.hist([defect_score, pass_score], color=['r','g'], label=['Defect','Pass'], **hist_para)
    plt.ylabel('Log(Count)')
    plt.xlabel('Probability')
    plt.xlim((0,1))
    plt.title('Summary Histogram')
    plt.legend()
    fig=plt.gcf()
    fig.set_facecolor('white')
    plt.savefig(save_path + 'Summary_distribution_log.jpg')
    plt.close()
    
    hist_para = {
        'bins'    : 100,
        'histtype': 'bar',
    }

    plt.hist(defect_score, color='red',**hist_para)
    plt.ylabel('Counts')
    plt.xlabel('Probability')
    plt.title('Defect Distrubution Histogram\n(Image Count : {})'.format(defect_score.shape[0]))
    fig=plt.gcf()
    fig.set_facecolor('white')
    plt.savefig(save_path + 'defect_distribution.jpg')
    plt.close()
    
    plt.hist(pass_score, color='g', **hist_para)
    plt.ylabel('Counts')
    plt.xlabel('Probability')
    plt.title('Pass Distrubution Histogram\n(Image Count : {})'.format(pass_score.shape[0]))
    fig=plt.gcf()
    fig.set_facecolor('white')
    plt.savefig(save_path + 'pass_distribution.jpg')
    plt.close()

f_p = open(master_path + '{}/pass.txt'.format(device_name),'r')
f_d = open(master_path + '{}/defect.txt'.format(device_name),'r')
pass_score = []
defect_score = []
for ps_score in f_p.readlines():
    pass_score.append(float(ps_score.split(',')[-1]))
for df_score in f_d.readlines():
    defect_score.append(float(df_score.split(',')[-1]))

visualize_distribution(defect_score,pass_score)
