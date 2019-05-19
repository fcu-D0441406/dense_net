import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2
from ops import *

ld = 10.0
img_size = 32
channel = 3
batch_size = 100
per_img = 10
n_dim = 64
lr = 0.00005
epoch = 50
gan_type = 'sagan'
ckpts = './checkpoint2_dir'
adam_meta = './checkpoint2_dir/MyModel'
output_dir = './output3'


#mnist = input_data.read_data_sets('./MNIST', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(per_img, per_img))
    gs = gridspec.GridSpec(per_img, per_img)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if(sample.shape[-1]==1):
            plt.imshow(sample.reshape(img_size, img_size), cmap='Greys_r')
        else:
            plt.imshow(sample.reshape(img_size, img_size,channel))

    return fig 

def batch_leaky(x,trainable=True):
    x = tf.contrib.layers.batch_norm(x,is_training=trainable)
    x = tf.nn.leaky_relu(x)
    return x

def batch_relu(x,trainable=True):
    x = tf.contrib.layers.batch_norm(x,is_training=trainable)
    x = tf.nn.relu(x)
    return x

def get_data(path):
    image_list = []
    for root,dirs,files in os.walk(path):
        for f in files:
            image_list.append(os.path.join(root,f))
    return image_list

def generator(z,reuse=False,trainable=True):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        ch = 256
        s = tf.split(z,2,axis=1)
        print(s)
        x = fully_conneted(z,4*4*ch)
        x = batch_relu(x,trainable)
        x = tf.reshape(x,(-1,4,4,ch))
        ch=ch//2
        print(x)
        deconv1 = risidual_up_block(x,s[0],ch,trainable,'up0')
        ch=ch//2
        print(deconv1)
        
        deconv2 = risidual_up_block(deconv1,s[1],ch,trainable,'up1')
        if(gan_type=='sagan'):
            deconv2 = attention(deconv2,ch,reuse=reuse)
        print(deconv2)
        
        #deconv3 = risidual_up_block(deconv2,s[3],1,trainable,'up2')
        #deconv3 = batch_relu(deconv3,trainable)
        #deconv3 = conv(deconv3, channels=1, kernel=3, stride=1, pad=1,scope='G_logit')

        deconv3 = risidual_up_block(deconv2,z,ch,trainable,scope='deconv3')
        deconv3 = conv(deconv3,channel,3,1,pad=1,scope='last_deconv')
        print(deconv3)
        
        x = tf.nn.tanh(deconv3)

        return x
    
def discriminator(image,reuse=False,trainable=True):
    with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
        ch = 64
        conv1 = risidual_down_block(image,ch,trainable,'down0')
        if(gan_type=='sagan'):
            conv1 = attention(conv1,ch,reuse=reuse)
        ch = ch*2
        print(conv1)
        
        conv2 = risidual_down_block(conv1,ch,trainable,'down1')
        ch = ch*2
        print(conv2)
        
        conv3 = risidual_down_block(conv2,ch,trainable,'down2')
        print(conv3)
        
        conv4 = conv(conv3,1,4,stride=1,scope='D_logit')
        #out= fully_conneted(conv4,1)
        #out = tf.sigmoid(conv4)
        print(conv4)
    return conv4


def risidual_up_block(x,z,ch,trainable,scope=''):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x2 = deconv(x,ch,3,2)
            x2 = condition_batch_norm(x2,z)
            x2 = tf.nn.relu(x2)
            
        with tf.variable_scope('res2'):
            x2 = deconv(x2,ch,3,1)
            x2 = condition_batch_norm(x2,z)
            x2 = relu(x2)
            
        x = deconv(x,ch,3,2)
        x = tf.add(x,x2)
        
    return x

def risidual_down_block(x,ch,trainable,scope):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x2 = conv(x,ch,3,2,pad=1)
            x2 = batch_leaky(x2,trainable)
            
        with tf.variable_scope('res2'):
            x2 = conv(x2,ch,3,1,pad=1)
            x2 = batch_leaky(x2,trainable)
        
        x = conv(x,ch,3,2,pad=1)

        x = tf.add(x,x2)
        
    return x

def condition_batch_norm(x, z, is_training=True, scope='batch_norm'):
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

        beta = fully_conneted(z, units=c, scope='beta')
        gamma = fully_conneted(z, units=c, scope='gamma')

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
            ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)

def attention(x, ch, sn=False, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        f = tf.layers.conv2d(x,ch//8,1,1)
        g = tf.layers.conv2d(x,ch//8,1,1)
        h = tf.layers.conv2d(x,ch,1,1)
        
        f = tf.reshape(f,(batch_size,-1,f.shape[-1]))
        g = tf.reshape(g,(batch_size,-1,g.shape[-1]))
        h = tf.reshape(h,(batch_size,-1,h.shape[-1]))
        
        s = tf.matmul(g,f,transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.matmul(beta,h)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o,shape=x.shape)
        o = conv(o, ch,1,1,pad=0,scope='attn_conv')
        x = gamma*o+x
        
    return x



def model_loss(Gz,Dr_logits,Df_logits):
    if(gan_type=='sagan'):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - Dr_logits))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + Df_logits))
        D_loss = real_loss+fake_loss
        
        G_loss = -tf.reduce_mean(Df_logits)
        
    else:
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dr_logits, labels=tf.ones_like(Dr_logits)))
        
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Df_logits, labels=tf.zeros_like(Df_logits)))
        D_loss = D_loss_real + D_loss_fake
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Df_logits, labels=tf.ones_like(Df_logits)))
    
    return D_loss,G_loss

def gradient_penalty(real, fake):
    if gan_type == 'drgan' :
        eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
        _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

        fake = real + 0.5 * x_std * eps

    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolated = real + alpha * (fake - real)

    logit = discriminator(interpolated, reuse=True)

    grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
    grad = tf.layers.flatten(grad)
    grad_norm = tf.norm(grad, axis=1)  # l2 norm

    GP = 0
    
    # WGAN - LP
    if gan_type == 'wgan':
        GP = ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
    '''
    else:
        GP = ld * tf.reduce_mean(tf.square(grad_norm - 1.))
    '''

    return GP

def optimizer(D_loss,G_loss):
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_opt = tf.train.AdamOptimizer(lr*4,beta1=0.0,beta2=0.9).minimize(D_loss, var_list=D_vars)
    
        G_opt = tf.train.AdamOptimizer(lr,beta1=0.0,beta2=0.9).minimize(G_loss, var_list=G_vars)
    return D_opt,G_opt

def check_enviroment():
    if not os.path.exists('output4/'):
        os.makedirs('output4/')
    if not os.path.exists(ckpts):
        os.makedirs(ckpts)

def noise_sample(batch_size,noise_dim):
    return np.random.uniform(0,0.5,size=[batch_size, noise_dim])

def gen_trainsform(img):
    img = (img+1)/2
    return img

def norm(img):
    img = (img/127.5)-1
    return img

def get_img(image_list,idx):
    img = []
    for ix in idx:
        img.append(cv2.imread(image_list[ix]))
    img = np.array(img)
    img = norm(img)
    img = np.reshape(img,(batch_size,img_size,img_size,channel))
    return img

if(__name__=='__main__'):
    check_enviroment()
    image_list = get_data('./img2')
    image_idx = np.arange(len(image_list))
    
    #noise_input = tf.placeholder(tf.float32,[batch_size,n_dim])
    #noise_input = 0.5 * tf.truncated_normal([batch_size, n_dim])
    noise_input = tf.multiply(tf.truncated_normal(shape=[batch_size,n_dim], name='random_z'),0.5)
    real_img = tf.placeholder(tf.float32,[batch_size,img_size,img_size,channel])
    
    Gz = generator(noise_input)
    Dr_logits = discriminator(real_img)
    Df_logits = discriminator(Gz,reuse=True)
    
    D_loss,G_loss = model_loss(Gz,Dr_logits,Df_logits)
    '''
    if(gan_type=='sagan'):
        GP = gradient_penalty(real_img, Gz)
        D_loss+=GP
    '''
    D_opt,G_opt = optimizer(D_loss,G_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'

        num = 0
        for i in range(epoch):
            np.random.shuffle(image_idx)
            for j in range(len(image_list)//batch_size):
                if(j%20==0):
                    g_noise = noise_sample(batch_size,n_dim)
                    G_sample = sess.run(Gz,feed_dict={noise_input:g_noise})
                    G_sample = gen_trainsform(G_sample)
                    fig = plot(G_sample)
                    plt.savefig('output3/{}.png'.format(str(num).zfill(3)), bbox_inches='tight')
                    num += 1
                    plt.close(fig)
            
                input_imgs = get_img(image_list,image_idx[j*batch_size:(j+1)*batch_size])
                '''
                _, d_loss = sess.run([D_opt, D_loss],
                                          feed_dict={real_img:input_imgs,noise_input:noise_sample(batch_size,n_dim)})
                
                _, g_loss = sess.run([G_opt, G_loss],
                                          feed_dict={real_img:input_imgs,noise_input: noise_sample(batch_size, n_dim)})
                '''
                _, d_loss = sess.run([D_opt, D_loss],
                                          feed_dict={real_img:input_imgs})
                
                _, g_loss = sess.run([G_opt, G_loss],
                                          feed_dict={real_img:input_imgs})
                
                print('step:',i,'d_loss:',d_loss,'g_loss:',g_loss)

            saver.save(sess,adam_meta,global_step=i) 
            
    
