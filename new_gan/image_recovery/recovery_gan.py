import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

lr = 0.0001

def upsample(x):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    return x

def generator(x):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        net = utils.conv(x,64,3,2,pad=1,scope='conv0')#128
        net = batch_relu(net)
        net1 = batch_relu(utils.conv(net,128,3,2,pad=1,scope='conv1'))#64
        net1 = batch_relu(utils.conv(net1,128,3,1,pad=1,scope='conv2'))
        net2 = batch_relu(utils.conv(net1,256,3,2,pad=1,scope='conv3'))#32
        net2 = batch_relu(utils.conv(net2,256,3,1,pad=1,scope='conv4'))
        net3 = batch_relu(utils.conv(net2,512,3,2,pad=1,scope='conv5'))#16
        net3 = batch_relu(utils.conv(net3,512,3,1,pad=1,scope='conv6'))
        
        net2_ = upsample(net3)
        net2_ = tf.concat([net2,net2_],axis=-1)
        net2_ = batch_leaky(utils.conv(net2_,256,3,1,pad=1,scope='up_conv1'))#32
        
        net1_ = upsample(net2_)
        net1_ = tf.concat([net1,net1_],axis=-1)
        net1_ = batch_leaky(utils.conv(net1_,128,3,1,pad=1,scope='up_conv2'))#64
        
        net_ = upsample(net1_)
        net_ = tf.concat([net,net_],axis=-1)
        net_ = batch_leaky(utils.conv(net_,64,3,1,pad=1,scope='up_conv3'))#128
        
        image = upsample(net_)
        image = tf.concat([x,image],axis=-1)
        image = utils.conv(image,3,3,1,pad=1,scope='conv_out')#128
        image = tf.nn.tanh(image)
        print(image)
    
    return image

def discriminator(image,reuse=tf.AUTO_REUSE):
    with tf.variable_scope('discriminator',reuse=reuse):
        conv1 = utils.conv(image,64,kernel=7,stride=2, pad=3,scope='d_conv1')
        conv1 = batch_leaky(conv1)
        print(conv1)
        conv2 = utils.conv(conv1,128,kernel=5,stride=2, pad=2,scope='d_conv2')
        conv2 = batch_leaky(conv2)
        print(conv2)
        conv3 = utils.conv(conv2,256,kernel=5,stride=2, pad=2,scope='d_conv3')
        conv3 = batch_leaky(conv3)
        print(conv3)
        
        conv4 = utils.conv(conv3,512,kernel=5,stride=2, pad=2,scope='d_conv4')
        conv4 = batch_leaky(conv4)
        print(conv4)
        
        #out = tf.reduce_sum(conv4,axis=[1,2])
        out = utils.conv(conv4,1,kernel=1,stride=1, pad=0,scope='out')
        print(out)
        
    return out

def model_loss(Dr_logits,Df_logits):
    '''
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dr_logits, labels=tf.ones_like(Dr_logits)))
    
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Df_logits, labels=tf.zeros_like(Df_logits)))
    D_loss = D_loss_real + D_loss_fake
    
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Df_logits, labels=tf.ones_like(Df_logits)))
    '''
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - Dr_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + Df_logits))
    D_loss = real_loss+fake_loss
    
    G_loss = -tf.reduce_mean(Df_logits)
    
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
original_img_ = np.zeros((8,256,256,3))
noise_img_ = np.zeros((8,256,256,3))
for i in range(8):
    original_img_[i] = original_img.copy()
    noise_img_[i] = noise_img.copy()
    
real_img = tf.placeholder(tf.float32,[None,256,256,3])
x = tf.placeholder(tf.float32,[None,256,256,3])

Gz = generator(x)
'''
Dr_logits = discriminator(real_img)
Df_logits = discriminator(Gz)
'''

Dr_logits = discriminator(tf.concat([x,real_img],axis=-1))
Df_logits = discriminator(tf.concat([x,Gz],axis=-1))

D_loss,G_loss = model_loss(Dr_logits,Df_logits)
cycle_gan = tf.reduce_mean(tf.abs(real_img-Gz))
G_loss = G_loss+5*cycle_gan
D_opt,G_opt = optimizer(D_loss,G_loss)


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


