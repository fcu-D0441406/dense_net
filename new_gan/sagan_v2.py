import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#from tensorflow.examples.tutorials.mnist import input_data
import cv2
from ops import *

ld = 10.0
img_size = 32
channel = 1
batch_size = 64
per_img = 8
n_dim = 64
lr = 0.0001
epoch = 50
gan_type = 'sagan'
tfrecord_train = './train.tfrecords'
ckpts = './checkpoint1_dir'
adam_meta = './checkpoint1_dir/MyModel'
output_dir = './output2'

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
def get_data(path):
    image_list = []
    for root,dirs,files in os.walk(path):
        for f in files:
            image_list.append(os.path.join(root,f))
    return image_list

def discriminator(x,trainable=True,reuse=tf.AUTO_REUSE):
    
    with tf.variable_scope("discriminator", reuse=reuse):
        ch = 64
        x = conv(x,32,3,1,pad=1,scope='inital_block')
        x = risidual_down_block(x,ch,2,trainable,'conv0')
        ch=ch*2
        print(x)
        x = risidual_down_block(x,ch,4,trainable,'conv1')
        x = attention(x,x.shape[-1])
        ch = ch*2
        print(x)
        x = risidual_down_block(x,ch,8,trainable,'conv2')
        print(x)
        #x = risidual_down_block(x,ch,16,trainable,'conv3') 
        #print(x)
        x = conv(x,1,4,stride=1,scope='D_logit')
        #x = tf.layers.flatten(x)
        #x = tf.layers.conv2d(x,1,4,1,padding='VALID')
        #x = tf.nn.sigmoid(x)
        print(x)
    return x

def generator(z,trainable=True, reuse=tf.AUTO_REUSE):
    #z = tf.reshape(z,(-1,1,1,n_dim))
    with tf.variable_scope("generator", reuse=reuse):
        ch = 256
        #z = tf.reshape(z,(batch_size,1,1,n_dim))
        #z = deconv(z,ch,4,4)
        z = fully_conneted(z,4*4*ch,scope='deconv_dense')
        z = batch_relu(z)
        z = tf.reshape(z,(-1,4,4,ch))
        ch = ch//2
        print(z)
        '''
        z = risidual_up_block(z,ch,trainable,scope='deconv0')
        ch = ch//2
        print(z)
        '''
        z = risidual_up_block(z,ch,4,trainable,scope='deconv1')
        ch = ch//2
        print(z)
        z = risidual_up_block(z,ch,2,trainable,scope='deconv2')
        z = attention(z,z.shape[-1])
        ch = ch//2
        print(z)
        #z = risidual_up_block(z,channel,trainable,scope='deconv3')
        #z = tf.layers.conv2d_transpose(z,channel,3,2,padding='SAME')
        #z = deconv(z,channel,4,2)
        z = risidual_up_block(z,ch,1,trainable,scope='deconv3')
        z = conv(z,channel,3,1,pad=1,scope='last_deconv')
        z = tf.nn.tanh(z)
        print(z)
    return z
    
def risidual_up_block(x,ch,r_num,trainable,scope=''):
    with tf.variable_scope(scope):
        '''
        with tf.variable_scope('res1'):
            x2 = deconv(x,ch,5,2)
            x2 = batch_leaky(x2,trainable)
        
        with tf.variable_scope('res2'):
            x2 = deconv(x2,ch,5,1)
            x2 = batch_leaky(x2,trainable)
            
        x = deconv(x,ch,5,2)
        x = tf.add(x,x2)
        '''
        resnext_b = []
        for i in range(r_num):
            with tf.variable_scope(str(i)):
                resnext_b.append(resnext_block(x,trainable))
        x2 = Concatenation(resnext_b)
        x2 = resnext_trainsition(x2,ch,trainable)
        x = deconv(x,ch,3,2)
        print(x,x2)
        x = tf.nn.relu(x+x2)
    return x

def resnext_block(x,trainable):
    x = deconv(x,32,3,2)
    x = batch_relu(x,trainable)
    x = conv(x,32,3,1,pad=1)
    x = batch_relu(x)
    return x

def resnext_trainsition(x,out_dim,trainable):
    x = tf.layers.conv2d(x,out_dim,1,1)
    x = batch_relu(x,trainable)
    #print(out_dim)
    return x

#d_num = [1,2,4,8]
def risidual_down_block(x,ch,d_num,trainable,scope):
    with tf.variable_scope(scope):
        '''
        with tf.variable_scope('res1'):
            x2 = conv(x,ch,5,2,pad=2)
            x2 = batch_leaky(x2,trainable)
        
        with tf.variable_scope('res2'):
            x2 = conv(x2,ch,5,1,pad=2)
            x2 = batch_leaky(x2,trainable)

        x = conv(x,ch,5,2,pad=2)

        x = tf.add(x,x2)
        
        '''
        x = dense_block(x,d_num,trainable,True)
        
        #x = res_block(x,ch,2,flag=True,trainable=True)
        
    return x

def dense_block(x,block_num,trainable,trainsition=True,scope=''):
    with tf.variable_scope(scope):
        for i in range(block_num):
            with tf.variable_scope(str(i)):
                x = bottleneck(x,trainable)
            
        if(trainsition==True):
            x = trainsition_layer(x,trainable)
    return x

def bottleneck(x,trainable):
    #print(x)
    with tf.variable_scope('dense1'):
        x2 = batch_leaky(x,trainable)
        x2 = conv(x2,16*2,1,1,pad=0)
    #x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
    with tf.variable_scope('dense2'):
        x2 = batch_leaky(x2,trainable)
        x2 = conv(x2,16,3,1,pad=1)
    
    #x2 = tf.layers.dropout(x2,self.drop_rate,training=trainable)
    #x2 = self.cbam_module(x2)
    x = Concatenation([x,x2])
    return x

def trainsition_layer(x,trainable):
    with tf.variable_scope('trainsition'):
        x = batch_leaky(x,trainable)
        C_channel = x.shape.as_list()[-1]
        #C_channel = int(C_channel*self.grow_rate)
        x = conv(x,C_channel,1,1,pad=0)
        #x = tf.layers.dropout(x,self.drop_rate,training=trainable)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        return x

    
'''
def res_block(x,out_dim,block_num,flag=True,trainable=True):
    
    for i in range(block_num):
        if(flag==False):
            stride = 1
        else:
            stride = 2
            channel = x.shape[-1]//2
        res_block = merge_block(x,stride,trainable)
        res_block = transition_layer(res_block,out_dim,trainable)
        
        if(flag):
            pre_block = tf.layers.average_pooling2d(x,2,2,padding='SAME')
            pre_block = tf.pad(pre_block,[[0,0],[0,0],[0,0],[channel,channel]])
            #pre_block = tf.layers.conv2d(x,out_dim,3,2,padding='SAME')
            flag = False
        else:
            pre_block = x
        #print(pre_block,res_block)
        #print(pre_block,res_block)
        x = pre_block+res_block
        x = tf.nn.relu(pre_block+res_block)
    return x
        
   
def transition_layer(x,out_dim,trainable):
    x = tf.layers.conv2d(x,out_dim,1,1)
    x = tf.layers.batch_normalization(x,training=trainable)
    #print(out_dim)
    return x

def merge_block(x,stride,trainable): 
    sp_block = list()
    for i in range(4):
        sp_block.append(bottleneck(x,stride,trainable))
    return Concatenation(sp_block)
        
def bottleneck(x,stride,trainable):
    
    x = tf.layers.conv2d(x,32,1,1,padding='SAME',
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    x = tf.layers.batch_normalization(x,training=trainable)
    x = tf.nn.relu(x)
   
    x = tf.layers.conv2d(x,32,3,stride,padding='SAME',
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    x = tf.layers.batch_normalization(x,training=trainable)
    x = tf.nn.relu(x)
    
    return x
'''
def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def batch_leaky(x,trainable=True):
    x = tf.layers.batch_normalization(x,training=trainable)
    x = tf.nn.leaky_relu(x)
    return x

def batch_relu(x,trainable=True):
    x = tf.layers.batch_normalization(x,training=trainable)
    x = tf.nn.relu(x)
    return x

def attention(x, ch, scope='attention', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
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

def optimizer(D_loss,G_loss):
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_opt = tf.train.AdamOptimizer(lr*4,beta1=0.0,beta2=0.9).minimize(D_loss, var_list=D_vars)
    
        G_opt = tf.train.AdamOptimizer(lr,beta1=0.0,beta2=0.9).minimize(G_loss, var_list=G_vars)
    return D_opt,G_opt


def check_enviroment():
    if not os.path.exists('output2/'):
        os.makedirs('output2/')
    if not os.path.exists(ckpts):
        os.makedirs(ckpts)

def noise_sample(batch_size,noise_dim):
    return np.random.uniform(-1., 1., size=[batch_size, noise_dim])

def gen_trainsform(img):
    img = (img+1)/2
    return img

def norm(img):
    img = (img/127.5)-1
    #print(img)
    return img

def get_img(image_list,idx):
    img = []
    for ix in idx:
        if(channel==1):
            img.append(cv2.imread(image_list[ix],0))
        else:
            img.append(cv2.imread(image_list[ix]))
    img = np.array(img)
    img = norm(img)
    img = np.reshape(img,(batch_size,img_size,img_size,channel))
    return img

if(__name__=='__main__'):
    check_enviroment()
    image_list = get_data('./img')
    image_idx = np.arange(len(image_list))
    
    noise_input = tf.placeholder(tf.float32,[batch_size,n_dim])
    #noise_input = tf.truncated_normal(shape=[batch_size,n_dim], name='random_z')
    real_img = tf.placeholder(tf.float32,[batch_size,img_size,img_size,channel])
    
    Gz = generator(noise_input)
    Dr_logits = discriminator(real_img)
    Df_logits = discriminator(Gz,reuse=True)
    
    D_loss,G_loss = model_loss(Gz,Dr_logits,Df_logits)

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
                    plt.savefig('output2/{}.png'.format(str(num).zfill(3)), bbox_inches='tight')
                    num += 1
                    plt.close(fig)
            
                input_imgs = get_img(image_list,image_idx[j*batch_size:(j+1)*batch_size])
                
                _, d_loss = sess.run([D_opt, D_loss],
                                          feed_dict={real_img:input_imgs,noise_input:noise_sample(batch_size,n_dim)})
                
                _, g_loss = sess.run([G_opt, G_loss],
                                          feed_dict={real_img:input_imgs,noise_input: noise_sample(batch_size, n_dim)})
                '''
                _, d_loss = sess.run([D_opt, D_loss],
                                          feed_dict={real_img:input_imgs})
                
                _, g_loss = sess.run([G_opt, G_loss],
                                          feed_dict={real_img:input_imgs})
                '''
                print('step:',i,'d_loss:',d_loss,'g_loss:',g_loss)

            saver.save(sess,adam_meta,global_step=i) 