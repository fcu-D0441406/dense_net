import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

ckpts = './checkpoint1_dir'
adam_meta = './checkpoint1_dir/MyModel'
epoch = 100
batch_size=8
img_size = 256
channel=3
lr = 2e-4

def get_data(img_path):
    img_files = []
    #print(img_path,type(img_path))
    for root,dirs,files in os.walk(img_path):
        for file in files:
            img_files.append(os.path.join(root,file))
    #print(img_files)
    return img_files

def generator_defect_to_pass(x,trainable=True):
    with tf.variable_scope("transfer_pass", reuse=tf.AUTO_REUSE):
        ch = 64
        x = tf.layers.conv2d(x,ch,7,1,padding='SAME')
        x = res_block(x,ch*2,2,trainable)#128
        #print(x)
        x = res_block(x,ch*4,2,trainable)#64
        #print(x)
        x = res_block(x,ch*4,1,trainable)#64
        #print(x)
        x = res_block(x,ch*4,1,trainable)#64
        #print(x)
        x = res_up_block(x,ch*2,trainable)#128
        #print(x)
        x = res_up_block(x,ch,trainable)#256
        #print(x)
        x = tf.layers.conv2d(x,channel,3,1,padding='SAME')
        x = tf.nn.tanh(x)
        print(x)
        return x
    

def generator_pass_to_defect(x,trainable=True):
    with tf.variable_scope("transfer_defect", reuse=tf.AUTO_REUSE):
        ch = 64
        x = tf.layers.conv2d(x,ch,7,1,padding='SAME')
        x = res_block(x,ch*2,2,trainable)#128
        #print(x)
        x = res_block(x,ch*4,2,trainable)#64
        #print(x)
        x = res_block(x,ch*4,1,trainable)#64
        #print(x)
        x = res_block(x,ch*4,1,trainable)#64
        #print(x)
        x = res_up_block(x,ch*2,trainable)#128
        #print(x)
        x = res_up_block(x,ch,trainable)#256
        #print(x)
        x = tf.layers.conv2d(x,channel,3,1,padding='SAME')
        x = tf.nn.tanh(x)
        print(x)
        return x

def discrimator_pass(x,trainable=True):
    with tf.variable_scope("discrimator_pass", reuse=tf.AUTO_REUSE):
        ch = 64
        x = tf.layers.conv2d(x,ch,7,2,padding='SAME')#128
        x = res_block(x,ch*2,2,trainable)#164
        #print(x)
        x = res_block(x,ch*4,2,trainable)#32
        #print(x)
        x = res_block(x,ch*8,2,trainable)#16
        #print(x)
        x = res_block(x,ch*16,2,trainable)#8
        #print(x)
        x = tf.layers.average_pooling2d(x,8,1,padding='VALID')
        x = tf.layers.dense(x,1)
        print(x)
        return x

def discrimator_defect(x,trainable=True):
    with tf.variable_scope("discrimator_defect", reuse=tf.AUTO_REUSE):
        ch = 64
        x = tf.layers.conv2d(x,ch,7,2,padding='SAME')#128
        x = res_block(x,ch*2,2,trainable)#164
        #print(x)
        x = res_block(x,ch*4,2,trainable)#32
        #print(x)
        x = res_block(x,ch*8,2,trainable)#16
        #print(x)
        x = res_block(x,ch*16,2,trainable)#8
        #print(x)
        x = tf.layers.average_pooling2d(x,8,1,padding='VALID')
        x = tf.layers.dense(x,1)
        print(x)
        return x

def res_block(x,ch,stride,trainable=True):
    short_cut = x
    x2 = tf.layers.conv2d(x,ch,3,stride,padding='SAME')
    x2 = batch_relu(x2,trainable)
    x2 = tf.layers.conv2d(x2,ch,3,1,padding='SAME')
    x2 = batch_relu(x2,trainable)
    if(stride==2):
        short_cut = tf.layers.conv2d(short_cut,ch,3,2,padding='SAME')
    if(short_cut.shape[-1]!=ch):
        short_cut = tf.layers.conv2d(short_cut,ch,3,1,padding='SAME')
    return tf.add(x2,short_cut)

def res_up_block(x,ch,trainable=True):
    short_cut = x
    x2 = tf.layers.conv2d_transpose(x,ch,3,2,padding='SAME')
    x2 = batch_relu(x2,trainable)
    x2 = tf.layers.conv2d(x2,ch,3,1,padding='SAME')
    x2 = batch_relu(x2,trainable)

    short_cut = tf.layers.conv2d_transpose(short_cut,ch,3,2,padding='SAME')
    
    return tf.add(x2,short_cut)

def batch_relu(x,trainable):
    x = tf.layers.batch_normalization(x,training=trainable)
    x = tf.nn.relu(x)
    return x

def discrimator_pass_loss(d_real_pass,d_fake_pass,cycle_loss):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_real_pass, labels=tf.ones_like(d_real_pass)))
        
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_pass, labels=tf.zeros_like(d_fake_pass)))
    D_loss = D_loss_real + D_loss_fake
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_pass, labels=tf.ones_like(d_fake_pass)))
    
    G_loss = G_loss+(cycle_loss*10)
    
    return D_loss,G_loss

def discrimator_defect_loss(d_real_defect,d_fake_defect,cycle_loss):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_real_defect, labels=tf.ones_like(d_real_defect)))
        
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_defect, labels=tf.zeros_like(d_fake_defect)))
    D_loss = D_loss_real + D_loss_fake
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_defect, labels=tf.ones_like(d_fake_defect)))
    
    G_loss = G_loss+(cycle_loss*10)
    
    return D_loss,G_loss

def cycle_loss(real_pass,real_defect,fake_pass_defect,fake_defect_pass):
    cycle_pass_loss = tf.reduce_mean(tf.abs(real_pass-fake_defect_pass))
    cycle_defect_loss = tf.reduce_mean(tf.abs(real_defect-fake_pass_defect))
    cycle_loss = cycle_pass_loss+cycle_defect_loss
    return cycle_loss

def d_optimizer(discrimator_pass_loss,discrimator_defect_loss):
    T_vars = tf.trainable_variables()
    Dp_vars = [var for var in T_vars if var.name.startswith('discrimator_pass')]
    Df_vars = [var for var in T_vars if var.name.startswith('discrimator_defect')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        Dp_opt = tf.train.AdamOptimizer(lr*4,beta1=0.0,beta2=0.9).minimize(discrimator_pass_loss, var_list=Dp_vars)
        
        Df_opt = tf.train.AdamOptimizer(lr*4,beta1=0.0,beta2=0.9).minimize(discrimator_defect_loss, var_list=Df_vars)
        #G_opt = tf.train.AdamOptimizer(lr,beta1=0.0,beta2=0.9).minimize(G_loss, var_list=G_vars)
    return Dp_opt,Df_opt


def g_optimizer(generator_pass_loss,generator_defect_loss):
    T_vars = tf.trainable_variables()
    Gp_vars = [var for var in T_vars if var.name.startswith('transfer_pass')]
    Gf_vars = [var for var in T_vars if var.name.startswith('transfer_defect')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        Gp_opt = tf.train.AdamOptimizer(lr,beta1=0.0,beta2=0.9).minimize(generator_pass_loss, var_list=Gp_vars)
        
        Gf_opt = tf.train.AdamOptimizer(lr,beta1=0.0,beta2=0.9).minimize(generator_defect_loss, var_list=Gf_vars)
    return Gp_opt,Gf_opt

def gen_trainsform(img):
    img = (img+1)/2
    return img

def norm(img):
    img = (img/127.5)-1
    #print(img)
    return img

def get_img(image_list,idx):
    #print(image_list)
    img = []
    for ix in idx:
        if(channel==1):
            img.append(cv2.imread(image_list[ix],0))
        else:
            img.append(cv2.imread(image_list[ix]))
    img = np.array(img)
    #print(img.shape)
    img = norm(img)
    img = np.reshape(img,(batch_size,img_size,img_size,channel))
    #print(img.shape)
    return img

def show_transfer_defect_img(sess,pass_img,real_pass,fake_defect):
    print(pass_img.shape)
    transfer_defect_img = np.array(sess.run([fake_defect],feed_dict={real_pass:pass_img}))[0]
    print(transfer_defect_img.shape)
    transfer_defect_img = gen_trainsform(transfer_defect_img)
    
    show_img(transfer_defect_img[0])

def show_img(img):
    if(img.shape[-1]==1):
        plt.imshow(img,cmap ='gray')
        plt.show()
    else:
        img = np.transpose(img,(2,1,0))
        plt.imshow(img)
        plt.show()

if(__name__=='__main__'):
    real_pass = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
    real_defect = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
    
    fake_pass = generator_defect_to_pass(real_defect)
    fake_pass_defect = generator_pass_to_defect(fake_pass)
    
    fake_defect = generator_pass_to_defect(real_pass)
    fake_defect_pass = generator_defect_to_pass(fake_defect)
    
    d_real_pass = discrimator_pass(real_pass)
    d_fake_pass = discrimator_pass(fake_pass)
    
    d_real_defect = discrimator_defect(real_defect)
    d_fake_defect = discrimator_defect(fake_defect)
    
    cycle_loss = cycle_loss(real_pass,real_defect,fake_pass_defect,fake_defect_pass)
    discrimator_pass_loss,generator_pass_loss = discrimator_pass_loss(d_real_pass,d_fake_pass,cycle_loss)
    discrimator_defect_loss,generator_defect_loss = discrimator_defect_loss(d_real_defect,d_fake_defect,cycle_loss)
    
    discrimator_pass_opt,discrimator_defect_opt = d_optimizer(discrimator_pass_loss,discrimator_defect_loss)
    generator_pass_opt,generator_defect_opt = g_optimizer(generator_pass_loss,generator_defect_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        pass_img_path = get_data('./train/0')
        pass_idx = np.arange(len(pass_img_path))
        defect_img_path = get_data('./train/1')
        defect_idx = np.arange(len(defect_img_path))
        
        saver = tf.train.Saver()
        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'
                
        for i in range(epoch):
                np.random.shuffle(pass_idx)
                np.random.shuffle(defect_idx)
                for j in range(len(pass_img_path)//batch_size):
                    pass_img = get_img(pass_img_path,pass_idx[j*batch_size:(j+1)*batch_size])
                    defect_img = get_img(defect_img_path,defect_idx[j*batch_size:(j+1)*batch_size])
                    
                    dp_loss,_ = sess.run([discrimator_pass_loss,discrimator_pass_opt],
                                         feed_dict={real_pass:pass_img,real_defect:defect_img})
                    df_loss,_ = sess.run([discrimator_defect_loss,discrimator_defect_opt],
                                         feed_dict={real_pass:pass_img,real_defect:defect_img})
                    
                    gp_loss,_ = sess.run([generator_pass_loss,generator_pass_opt],
                                         feed_dict={real_pass:pass_img,real_defect:defect_img})
                    gf_loss,_ = sess.run([generator_defect_loss,generator_defect_opt],
                                         feed_dict={real_pass:pass_img,real_defect:defect_img})
                    
                    if(j==0):
                        print(dp_loss,df_loss,gp_loss,gf_loss)
                        show_transfer_defect_img(sess,pass_img,real_pass,fake_defect)
                        saver.save(sess,adam_meta,global_step=i) 
    
    
    