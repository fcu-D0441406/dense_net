import tensorflow as tf
import numpy as np
import cv2
import os
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weight_init = tf.contrib.layers.variance_scaling_initializer()
style_num = 2
style_dim = 32
img_size = 64


def get_data_path(path):
    img_path_list = []
    label_list = []
    for root,dirs,files in os.walk(path):
        for d in dirs:
            for file in os.listdir(os.path.join(root,d)):
                f = os.path.join(os.path.join(root,d),file)
                img_path_list.append(f)
                label_list.append(d)
    return np.array(img_path_list),np.array(label_list)


def mapping_network(z,scope='mapping_network'):
    mapping_network_deep = 4
    ch = 32
    style_list = list()
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        for i in range(mapping_network_deep):
            z = tf.layers.dense(z,ch)
            z = tf.nn.relu(z)
            ch = ch*2

        for i in range(style_num) :
            style = tf.layers.dense(z,style_dim)
            style_list.append(style)
        return style_list

def style_encoder(input_img,scope='style_encoder'):
    ch = 16
    style_encoder_list = list()
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(input_img,ch,1,1,padding='SAME')
        
        for i in range(4):
            x = resblock_no_norm(x,ch,3,1,scope_name='discriminator_res_block'+str(i))
            x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
            ch = ch*2
            
        x = tf.nn.leaky_relu(x,0.2)
        x = tf.layers.conv2d(x,ch,kernel_size=4,strides=1,padding='VALID')
        x = tf.nn.leaky_relu(x,0.2)
        x = tf.layers.flatten(x)

        for i in range(style_num):
            style_encoder_list.append(tf.layers.dense(x,style_dim))
        
        return style_encoder_list

def generator(input_img,style_code,scope='generator'):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        ch = 32
        x = tf.layers.conv2d(input_img,ch,1,1,padding='SAME',kernel_initializer=weight_init)
        for i in range(4):
            x = resblock(x,ch,3,1,scope_name='resblock_down'+str(i))
            x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
            ch = ch*2
        
        for i in range(2):
            x = resblock(x,ch,3,1,scope_name='resblock'+str(i))
        
        for i in range(2):
            gamma1 = tf.layers.dense(style_code,ch)
            beta1 = tf.layers.dense(style_code,ch)
            
            gamma2 = tf.layers.dense(style_code,ch)
            beta2 = tf.layers.dense(style_code,ch)
            
            x = resblock_adain(x,ch,3,1,tf.reshape(gamma1,[gamma1.shape[0],1,1,-1]),tf.reshape(beta1,[beta1.shape[0],1,1,-1]),
                               tf.reshape(gamma2,[gamma2.shape[0],1,1,-1]),tf.reshape(beta2,[beta2.shape[0],1,1,-1]),
                               scope_name='ada_resblock'+str(i))
            
        for i in range(4):
            
            
            x = tf.image.resize_nearest_neighbor(x,[x.shape[1]*2,x.shape[2]*2])
            gamma1 = tf.layers.dense(style_code,ch)
            beta1 = tf.layers.dense(style_code,ch)
            gamma2 = tf.layers.dense(style_code,ch//2)
            beta2 = tf.layers.dense(style_code,ch//2)
            
            x = resblock_adain(x,ch//2,3,1,tf.reshape(gamma1,[gamma1.shape[0],1,1,-1]),tf.reshape(beta1,[beta1.shape[0],1,1,-1]),
                               tf.reshape(gamma2,[gamma2.shape[0],1,1,-1]),tf.reshape(beta2,[beta2.shape[0],1,1,-1]),
                               scope_name='ada_resblock_up'+str(i))
            ch = ch//2
        x = tf.layers.conv2d(x,3,1,1)
        #
        return tf.nn.sigmoid(x)
    
def discriminator(input_img,scope='discriminator'):
    ch = 32
    discriminator_list = list()
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(input_img,ch,1,1,padding='SAME')
        
        for i in range(4):
            x = resblock_no_norm(x,ch,3,1,scope_name='discriminator_res_block'+str(i))
            x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
            ch = ch*2
            
        x = tf.nn.leaky_relu(x,0.2)
        x = tf.layers.conv2d(x,ch,kernel_size=4,strides=1,padding='SAME')
        x = tf.nn.leaky_relu(x,0.2)
        
        x = tf.layers.flatten(x)
        
        for i in range(style_num):
            discriminator_list.append(tf.layers.dense(x,1))
        
        return discriminator_list



def discriminator_loss(real_logit,fake_logit):
    real_dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,labels=tf.ones_like(real_logit))
    fake_dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,labels=tf.zeros_like(fake_logit))
    
    return tf.reduce_mean(real_dis_loss+fake_dis_loss)

def generator_loss(fake_logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=tf.ones_like(fake_logits)))

def cycle_loss(original_img,cycle_original_img):
    return tf.reduce_mean(tf.abs(original_img-cycle_original_img))

def diverse_loss(fake_img1,fake_img2):
    return tf.reduce_mean(tf.abs(fake_img1-fake_img2))

def style_loss(target_style_encoder,random_style_encoder):
    return tf.reduce_mean(tf.abs(random_style_encoder-target_style_encoder))
        

def get_data_path(path,index):
    img_path_list = []
    label_list = []
    for file in os.listdir(path):
        f = os.path.join(path,file)
        img_path_list.append(f)
        label_list.append(index)
    return np.array(img_path_list),np.array(label_list)

def get_train_with_aug(img_list,label_list,shuffle_list,j):
    image = np.zeros((1,img_size,img_size,3),dtype=np.float32)
    label = 0
    now = 0
    for i in range(1*j,1*(j+1)):
        img = mpimg.imread(img_list[shuffle_list[i]])
        img = cv2.resize(img,(img_size,img_size))
        img = img/255.0
        '''
        random_rotate = np.random.randint(4)
        random_flip = np.random.randint(3)
        if(random_flip!=2):
            img = np.flip(img,random_flip)
        img = np.rot90(img,random_rotate)
        '''
        image[now] = img
        label = label_list[j]
        now+=1
    return image,label


def train():

    original_img = tf.placeholder(tf.float32,[1,img_size,img_size,3])
    target_img = tf.placeholder(tf.float32,[1,img_size,img_size,3])
    original_label = tf.placeholder(tf.int32,[])
    target_label = tf.placeholder(tf.int32,[])
    
    random_z = tf.random_normal([1,style_dim])
    random_style_encoder = tf.gather(mapping_network(random_z),target_label)
    
    random_z2 = tf.random_normal([1,style_dim])
    random_style_encoder2 = tf.gather(mapping_network(random_z2),target_label)
    random_z3 = tf.random_normal([1,style_dim])
    random_style_encoder3 = tf.gather(mapping_network(random_z3),target_label)

    fake_img = generator(original_img,random_style_encoder)
    fake_img2 = generator(original_img,random_style_encoder2)
    fake_img3 = generator(original_img,random_style_encoder3)
    
    original_style_encoder = tf.gather(style_encoder(original_img),original_label)
    fake_style_encoder = tf.gather(style_encoder(fake_img),target_label)
    target_style_encoder = tf.gather(style_encoder(target_img),target_label)
    
    
    
    cycle_original_img = generator(fake_img,original_style_encoder)
    fake_img_style_encoder = style_encoder(fake_img)
    
    real_logit = tf.gather(discriminator(original_img),original_label)
    fake_logit = tf.gather(discriminator(fake_img),target_label)
    
    generator_ls = generator_loss(fake_img)
    cycle_ls = cycle_loss(original_img,cycle_original_img)
    style_ls = style_loss(fake_style_encoder,random_style_encoder)
    diverse_ls = diverse_loss(fake_img2,fake_img3)
    
    g_loss = generator_ls+cycle_ls+style_ls-diverse_ls
    
    d_loss = discriminator_loss(real_logit,fake_logit)
    
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'generator' in var.name]
    e_vars = [var for var in t_vars if 'style_encoder' in var.name]
    f_vars = [var for var in t_vars if 'mapping_network' in var.name]
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    
    
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    
    with tf.name_scope('opt'):
        g_opt = tf.train.AdamOptimizer(1e-3,name='g_optimizer',beta1=0.0,beta2=0.99).minimize(g_loss,var_list=g_vars)
        e_opt = tf.train.AdamOptimizer(1e-3,name='e_optimizer',beta1=0.0,beta2=0.99).minimize(g_loss,var_list=e_vars)
        f_opt = tf.train.AdamOptimizer(1e-4*0.01,name='f_optimizer',beta1=0.0,beta2=0.99).minimize(g_loss,var_list=f_vars)
        d_opt = tf.train.AdamOptimizer(1e-3,name='d_optimizer',beta1=0.0,beta2=0.99).minimize(d_loss,var_list=d_vars)
        with tf.control_dependencies([g_opt, e_opt, f_opt]):
            g_opt = ema.apply(g_vars)
            e_opt = ema.apply(e_vars)
            f_opt = ema.apply(f_vars)


    dog_image_list,dog_label_list = get_data_path('./img/1',1)
    dog_shuffle_list = np.arange(dog_image_list.shape[0])
    
    cat_image_list,cat_label_list = get_data_path('./img/0',0)
    cat_shuffle_list = np.arange(cat_image_list.shape[0])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('start train')
        for i in range(1000):
            np.random.shuffle(dog_image_list)
            np.random.shuffle(cat_image_list)
            for j in range(dog_image_list.shape[0]):
                dog_img,dog_label = get_train_with_aug(dog_image_list,dog_label_list,dog_shuffle_list,j)
                cat_img,cat_label = get_train_with_aug(cat_image_list,cat_label_list,cat_shuffle_list,j)
                if(j%2==0):
                    sess.run(d_opt,feed_dict={original_img:dog_img,original_label:dog_label,
                                                                 target_img:cat_img,target_label:cat_label})
                    sess.run([g_opt,e_opt,f_opt],feed_dict={original_img:dog_img,original_label:dog_label,
                                                                 target_img:cat_img,target_label:cat_label})
                else:
                    sess.run(d_opt,feed_dict={original_img:dog_img,original_label:dog_label,
                                                                 target_img:cat_img,target_label:cat_label})
                    sess.run([g_opt,e_opt,f_opt],feed_dict={original_img:dog_img,original_label:dog_label,
                                                                 target_img:cat_img,target_label:cat_label})
    
            dog_img,dog_label = get_train_with_aug(dog_image_list,dog_label_list,dog_shuffle_list,0)
            cat_img,cat_label = get_train_with_aug(cat_image_list,cat_label_list,cat_shuffle_list,0)
            if(i%2==0):
                g_ls1,g_ls2,g_ls3,g_ls4,d_loss_,fake_img_,cy_img = sess.run([generator_ls,cycle_ls,style_ls,diverse_ls,d_loss,fake_img,cycle_original_img],
                                                     feed_dict={original_img:dog_img,original_label:dog_label,
                                                                target_img:cat_img,target_label:cat_label})
            else:
                g_ls1,g_ls2,g_ls3,g_ls4,d_loss_,fake_img_,cy_img = sess.run([generator_ls,cycle_ls,style_ls,diverse_ls,d_loss,fake_img,cycle_original_img],
                                                     feed_dict={original_img:cat_img,original_label:cat_label,
                                                                target_img:dog_img,target_label:dog_label})
            print(i,'generator_loss,cycle_loss,style_loss,diverse_loss',g_ls1,g_ls2,g_ls3,g_ls4,'d_loss',d_loss_)
            
            plt.imshow(dog_img[0])
            plt.show()
            plt.imshow(fake_img_[0])
            plt.show()
            plt.imshow(cy_img[0])
            plt.show()

    
train()
    

'''
with tf.control_dependencies([g_opt,e_opt,f_opt]):
    g_grads = g_opt.compute_gradients()
    e_grads = e_opt.compute_gradients(g_loss,e_vars)
    f_grads = f_opt.compute_gradients(g_loss,f_vars)
    d_grads = d_opt.compute_gradients(d_loss,d_vars)
    g_train_op = opt.apply_gradients(g_grads, global_step=global_step)
    e_train_op = opt.apply_gradients(e_grads, global_step=global_step)
    f_train_op = opt.apply_gradients(f_grads, global_step=global_step)
    d_train_op = opt.apply_gradients(d_grads, global_step=global_step)
'''