import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import matplotlib.image as mpimg
import cv2
from network import *

#stage=5
a = 4
is_continue = True
img_size = [4,8,16,32,64]
now_stage = [1,2,3,4,5]
epoch = [10,10,30,50,100]
adam_meta = './checkpoint1_dir/MyModel'
batch_size = 16

def get_data_path(path):
    img_path_list = []
    for f in os.listdir(path):
        f = os.path.join(path,f)
        img_path_list.append(f)
    return np.array(img_path_list)

def show_img(img):
    img = (img+1)*127.5
    img = np.array(img,np.uint8)
    plt.imshow(img)
    plt.show()

def get_batch_img(img_list,shuffle_list,j):
    image = np.zeros((batch_size,img_size[a],img_size[a],3),dtype=np.float32)
    now = 0
    for i in range(batch_size*j,batch_size*(j+1)):
        img = mpimg.imread(img_list[shuffle_list[i]])
        img = cv2.resize(img,(img_size[a],img_size[a]))
        img = (img/127.5)-1
        image[now] = img
        now+=1
    return image


def get_loss(real_logit,fake_logit):
    #d_loss = -1*tf.reduce_mean(real_logit)+tf.reduce_mean(fake_logit)
    d_real_ls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,labels=tf.ones_like(real_logit)))
    d_fake_ls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,labels=tf.zeros_like(fake_logit)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,labels=tf.ones_like(fake_logit)))
    d_loss = d_real_ls+d_fake_ls
    #g_loss = -1*tf.reduce_mean(fake_logit)
    
    return d_loss,g_loss

def gradient_penalty_loss(real_images,fake_images,stage,is_s):
    differences = fake_images - real_images
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolates = real_images + (alpha * differences)
    discri_logits= discriminator(interpolates,stage,is_s)
    gradients = tf.gradients(discri_logits, [interpolates])[0]

    # 2 norm
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty




data_path = get_data_path('./img')
data_shuffle_list = np.arange(data_path.shape[0])
print(data_shuffle_list.shape)
alpha = tf.placeholder(dtype=tf.float32,shape=[])
input_x = [tf.placeholder(dtype=tf.float32,shape=[None,4,4,3]),
           tf.placeholder(dtype=tf.float32,shape=[None,8,8,3]),
           tf.placeholder(dtype=tf.float32,shape=[None,16,16,3]),
           tf.placeholder(dtype=tf.float32,shape=[None,32,32,3]),
           tf.placeholder(dtype=tf.float32,shape=[None,64,64,3])]





z = tf.random_normal(shape=[batch_size,256])
if(img_size[a]>4):
    ###############################
    '''
    old saver
    '''
    fake_img = generator(z,now_stage[a-1],alpha)
    real_logit = discriminator(input_x[a-1],now_stage[a-1],alpha)
    fake_logit = discriminator(fake_img,now_stage[a-1],alpha)
    
    old_t_vars = tf.trainable_variables()
    old_g_vars = [var for var in old_t_vars if 'generator' in var.name]
    old_d_vars = [var for var in old_t_vars if 'discriminator' in var.name]
    old_vars = old_g_vars+old_d_vars
    
    old_saver = tf.train.Saver(var_list=old_vars)
    ################################

fake_img = generator(z,now_stage[a],alpha)
real_logit = discriminator(input_x[a],now_stage[a],alpha)
fake_logit = discriminator(fake_img,now_stage[a],alpha)

d_loss,g_loss = get_loss(real_logit,fake_logit)

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'generator' in var.name]
d_vars = [var for var in t_vars if 'discriminator' in var.name]
new_vars = g_vars+d_vars
g_opt = tf.train.AdamOptimizer(5e-4,name='g_optimizer',beta1=0.0,beta2=0.99).minimize(g_loss,var_list=g_vars)
d_opt = tf.train.AdamOptimizer(5e-4,name='d_optimizer',beta1=0.0,beta2=0.99).minimize(d_loss,var_list=d_vars)

saver = tf.train.Saver(var_list=new_vars)
total_step = epoch[a]*(data_path.shape[0]//batch_size)
increase_alpha = 1.0/(total_step//4)

alpha_value = 1.0
if(img_size[a]==4 or is_continue):
    alpha_value = 1.0


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if(is_continue):
        saver.restore(sess,'./checkpoint1_dir/MyModel')
    elif(img_size[a]>4 and is_continue==False):
        old_saver.restore(sess,'./checkpoint1_dir/MyModel')
    
    for i in range(epoch[a]):
        np.random.shuffle(data_shuffle_list)
        for j in range(data_path.shape[0]//batch_size):
            batch_x = get_batch_img(data_path,data_shuffle_list,j)
            _,d_ls = sess.run([d_opt,d_loss],feed_dict={input_x[a]:batch_x,alpha:alpha_value})
            
            _,g_ls = sess.run([g_opt,g_loss],feed_dict={input_x[a]:batch_x,alpha:alpha_value})
            
            if(alpha_value>=1.0):
                alpha_value = 1.0
            else:
                alpha_value+=increase_alpha
                
            if(j%100==0):
                print('d loss',d_ls,'g loss',g_ls)
        
        if(i%1==0):
            fake_img_,real_img_ = sess.run([fake_img,input_x[a]],feed_dict={input_x[a]:batch_x,alpha:alpha_value})
            fake_img_ = np.clip(fake_img_,-1,1)
            show_img(fake_img_[0])
            show_img(batch_x[0])
            
            saver.save(sess,adam_meta)
        
    
    
    
    
    
    
    
    