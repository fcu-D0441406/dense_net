import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2

ld = 10.0
img_size = 32
channel = 1
batch_size = 64
n_dim = 128
stride = 5
lr = 0.0001
gan_type = 'sagan'
tfrecord_train = './train.tfrecords'
ckpts = './checkpoint1_dir'
output_dir = './output2'
adam_meta = './checkpoint1_dir/MyModel'
#mnist = input_data.read_data_sets('./MNIST', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size), cmap='Greys_r')

    return fig

def read_and_decode(tfrecord_file_path,batch_size):
    tfrecord_file = tf.train.string_input_producer([tfrecord_file_path])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(tfrecord_file)
    img_features = tf.parse_single_example(serialized_example,features={
                                        'label':tf.FixedLenFeature([],tf.int64),
                                        'image_raw':tf.FixedLenFeature([],tf.string),
                                        })
    image = tf.decode_raw(img_features['image_raw'],tf.uint8)
    image = tf.reshape(image,[img_size,img_size,channel])
    label = tf.cast(img_features['label'],tf.int32)
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=batch_size,
                                                     min_after_dequeue=100,
                                                     num_threads=64,
                                                     capacity=200)
    return image_batch,tf.reshape(label_batch,[batch_size]) 

def batch_leaky(x,trainable=True):
    x = tf.contrib.layers.batch_norm(x,is_training=trainable)
    x = tf.nn.leaky_relu(x)
    return x

def batch_relu(x,trainable=True):
    x = tf.contrib.layers.batch_norm(x,is_training=trainable)
    x = tf.nn.relu(x)
    return x

def generator(z,reuse=False,trainable=True):
    with tf.variable_scope('generator',reuse=reuse):
        z = tf.reshape(z,(-1,1,1,n_dim))
        deconv1 = tf.layers.conv2d_transpose(z,512,4,1,padding='VALID')
        deconv1 = batch_leaky(deconv1,trainable)
        print(deconv1)
        deconv2 = tf.layers.conv2d_transpose(deconv1,256,4,2,padding='SAME')
        deconv2 = batch_leaky(deconv2,trainable)
        if(gan_type=='sagan'):
            deconv2 = attention(deconv2,256,reuse=reuse)
        print(deconv2)
        deconv3 = tf.layers.conv2d_transpose(deconv2,128,4,2,padding='SAME')
        deconv3 = batch_leaky(deconv3,trainable)
        print(deconv3)
        deconv4 = tf.layers.conv2d_transpose(deconv3,1,4,2,padding='SAME')
        deconv4 = tf.tanh(deconv4)
        print(deconv4)
    return deconv4

def attention(x, ch, sn=False, scope='attention', reuse=False):
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
        x = gamma*o+x
        
    return x

def discriminator(image,reuse=False,trainable=True):
    with tf.variable_scope('discriminator',reuse=reuse):
        conv1 = tf.layers.conv2d(image,128,4,2,padding='SAME')
        conv1 = batch_leaky(conv1,trainable)
        print(conv1)
        conv2 = tf.layers.conv2d(conv1,256,4,2,padding='SAME')
        conv2 = batch_leaky(conv2,trainable)
        if(gan_type=='sagan'):
            conv2 = attention(conv2,256,reuse=reuse)
        print(conv2)
        conv3 = tf.layers.conv2d(conv2,512,4,2,padding='SAME')
        conv3 = batch_leaky(conv3,trainable)
        print(conv3)
        conv4 = tf.layers.conv2d(conv3,1,4,1,padding='VALID')
        out = tf.sigmoid(conv4)
        print(conv4)
    return out,conv4
'''
def residual_block(x,trainable):
    res_x = tf.layers.conv2d(x,x.shape[-1],3,1,padding='SAME')
    res_x = batch_relu(res_x,trainable)
    
    res_x = tf.layers.conv2d(res_x,res_x.shape[-1],3,1,padding='SAME')
    res_x = batch_relu(res_x,trainable)
    
    return tf.add(res_x,x)

def up_residual_block(x,trainable):
    up_res_x
'''
def model_loss(Gz,Dr,Dr_logits,Df,Df_logits):
    if(gan_type=='w_gan'):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - Dr_logits))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + Df_logits))
        D_loss = real_loss+fake_loss
        
        G_loss = -tf.reduce_mean(Df_logits)
        
    else:
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dr_logits, labels=tf.ones_like(Dr)))
        
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Df_logits, labels=tf.zeros_like(Df)))
        D_loss = D_loss_real + D_loss_fake
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Df_logits, labels=tf.ones_like(Df)))
    
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
        D_opt = tf.train.AdamOptimizer(lr*4,beta1=0.5).minimize(D_loss, var_list=D_vars)
    
        G_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(G_loss, var_list=G_vars)
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
    return img

def get_img(sess,batch_test_x):
    img = sess.run(batch_test_x)
    img = norm(img)
    #print(img[0][0][0])
    return img
    '''
    mnist_img, _ = mnist.train.next_batch(batch_size)
    mnist_img = np.reshape(mnist_img,(-1,28,28,1))
    return mnist_img
    '''

if(__name__=='__main__'):
    check_enviroment()
    train_image,train_label = read_and_decode(tfrecord_train,batch_size)
    noise_input = tf.placeholder(tf.float32,[batch_size,n_dim])
    real_img = tf.placeholder(tf.float32,[batch_size,img_size,img_size,channel])
    
    Gz = generator(noise_input)
    Dr,Dr_logits = discriminator(real_img)
    Df,Df_logits = discriminator(Gz,reuse=True)
    
    D_loss,G_loss = model_loss(Gz,Dr,Dr_logits,Df,Df_logits)
    if(gan_type=='sagan'):
        GP = gradient_penalty(real_img, Gz)
        D_loss+=GP
    
    D_opt,G_opt = optimizer(D_loss,G_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        num = 402
        for i in range(100000):
            if(i%20==0):
                
                g_noise = noise_sample(batch_size,n_dim)
                G_sample = sess.run(Gz,feed_dict={noise_input:g_noise})
                G_sample = gen_trainsform(G_sample)
                fig = plot(G_sample)
                plt.savefig('output2/{}.png'.format(str(num).zfill(3)), bbox_inches='tight')
                num += 1
                plt.close(fig)
            
            input_imgs = get_img(sess,train_image)
            
            _, d_loss = sess.run([D_opt, D_loss],
                                      feed_dict={real_img:input_imgs,noise_input:noise_sample(batch_size,n_dim)})
            
            _, g_loss = sess.run([G_opt, G_loss],
                                      feed_dict={real_img:input_imgs,noise_input: noise_sample(batch_size, n_dim)})
            
            print('step:',i,'d_loss:',d_loss,'g_loss:',g_loss)
            if(i%100==0 and i!=0):
                saver.save(sess,adam_meta,global_step=i) 
            
        coord.request_stop()
        coord.join(threads)