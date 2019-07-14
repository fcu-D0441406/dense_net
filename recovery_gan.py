import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

lr = 0.0001

def generator(x):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        net = tf.layers.conv2d(x,64,3,2,padding='SAME')#128
        net = batch_relu(net)
        net1 = batch_relu(tf.layers.conv2d(net,128,3,2,padding='SAME'))#64
        net1 = batch_relu(tf.layers.conv2d(net1,128,3,1,padding='SAME'))
        net2 = batch_relu(tf.layers.conv2d(net1,256,3,2,padding='SAME'))#32
        net2 = batch_relu(tf.layers.conv2d(net2,256,3,1,padding='SAME'))
        net3 = batch_relu(tf.layers.conv2d(net2,512,3,2,padding='SAME'))#16
        net3 = batch_relu(tf.layers.conv2d(net3,512,3,1,padding='SAME'))
        
        net2_ = batch_relu(tf.layers.conv2d_transpose(net3,256,3,2,padding='SAME'))
        net2_ = tf.concat([net2,net2_],axis=-1)
        net2_ = batch_relu(tf.layers.conv2d(net2_,256,3,1,padding='SAME'))
        
        net1_ = batch_relu(tf.layers.conv2d_transpose(net2_,128,3,2,padding='SAME'))
        net1_ = tf.concat([net1,net1_],axis=-1)
        net1_ = batch_relu(tf.layers.conv2d(net1_,128,3,1,padding='SAME'))
        
        net_ = batch_relu(tf.layers.conv2d_transpose(net1_,64,3,2,padding='SAME'))
        net_ = tf.concat([net,net_],axis=-1)
        net_ = batch_relu(tf.layers.conv2d(net_,64,3,1,padding='SAME'))
        
        image = tf.layers.conv2d_transpose(net_,3,3,2,padding='SAME')
        image = tf.nn.tanh(image)
        print(image)
    
    return image

def discriminator(image,reuse=tf.AUTO_REUSE):
    with tf.variable_scope('discriminator',reuse=reuse):
        conv1 = tf.layers.conv2d(image,64,4,4,padding='SAME')#64
        conv1 = batch_leaky(conv1)
        print(conv1)
        conv2 = tf.layers.conv2d(conv1,128,4,2,padding='SAME')#32
        conv2 = batch_leaky(conv2)
        print(conv2)
        conv3 = tf.layers.conv2d(conv2,256,4,2,padding='SAME')#16
        conv3 = batch_leaky(conv3)
        print(conv3)
        
        conv4 = tf.layers.conv2d(conv3,512,4,4,padding='SAME')#4
        conv4 = batch_leaky(conv4)
        print(conv4)
        
        out = tf.layers.conv2d(conv4,1,4,1,padding='VALID')
        print(out)
        
    return out

def model_loss(Dr_logits,Df_logits):
    
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dr_logits, labels=tf.ones_like(Dr_logits)))
    
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Df_logits, labels=tf.zeros_like(Df_logits)))
    D_loss = D_loss_real + D_loss_fake
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Df_logits, labels=tf.ones_like(Df_logits)))
    
    return D_loss,G_loss

def optimizer(D_loss,G_loss):
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_opt = tf.train.AdamOptimizer(lr*4,beta1=0.5).minimize(D_loss, var_list=D_vars)
    
        G_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(G_loss, var_list=G_vars)
    return D_opt,G_opt
    

def batch_relu(x):
    return tf.layers.batch_normalization(tf.nn.relu(x))

def batch_leaky(x):
    return tf.layers.batch_normalization(tf.nn.leaky_relu(x))

def show_img(img):
    plt.imshow(img)
    plt.show()

def gen_trainsform(img):
    img = (img+1)/2
    return img

def img_norm(img):
    img = (img/127.5)-1
    return img


original_img = cv2.cvtColor(cv2.imread('./train18.jpg'),cv2.COLOR_BGR2RGB)

noise_img = original_img.copy()
show_img(original_img)
noise_img[70:160,100:180,:] = [255,255,255]

original_img = img_norm(original_img[np.newaxis,:,:,:])
noise_img = img_norm(noise_img[np.newaxis,:,:,:])
original_img_ = np.zeros((16,256,256,3))
noise_img_ = np.zeros((16,256,256,3))
for i in range(16):
    original_img_[i] = original_img.copy()
    noise_img_[i] = noise_img.copy()
    
real_img = tf.placeholder(tf.float32,[None,256,256,3])
x = tf.placeholder(tf.float32,[None,256,256,3])

Gz = generator(x)
Dr_logits = discriminator(real_img)
Df_logits = discriminator(Gz)

D_loss,G_loss = model_loss(Dr_logits,Df_logits)
D_opt,G_opt = optimizer(D_loss,G_loss)
cycle_gan = tf.reduce_mean(tf.abs(real_img-Gz))
G_loss = G_loss+cycle_gan

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    '''
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(ckpts) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(ckpts))
    else:
        assert 'can not find checkpoint folder path!'
    '''
    for i in range(1000):
            
        _, d_loss = sess.run([D_opt, D_loss],
                                  feed_dict={real_img:original_img_,x:noise_img_})
        
        _, g_loss = sess.run([G_opt, G_loss],
                                  feed_dict={real_img:original_img_,x:noise_img_})
        
        if(i%10==0):
            gz = sess.run(Gz,feed_dict={x:noise_img})
            gz = gen_trainsform(gz[0])
            show_img(gz)
        
        print('step:',i,'d_loss:',d_loss,'g_loss:',g_loss)


