import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2

ld = 10.0
img_size = 64
channel = 3
batch_size = 100
n_dim = 64
lr = 0.0001
gan_type = 'sagan'
tfrecord_train = './train.tfrecords'
ckpts = './checkpoint1_dir'
output_dir = './output2'
adam_meta = './checkpoint1_dir/MyModel'
#mnist = input_data.read_data_sets('./MNIST', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size,channel))
    plt.show()
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

def discriminator(x,training=True,reuse=tf.AUTO_REUSE):
    
    with tf.variable_scope("discriminator", reuse=reuse):
        x = conv(x,64,scope='conv1')
        x = batch_lrelu(x)
        print(x)
        x = conv(x,128,scope='conv2')
        x = batch_lrelu(x)
        print(x)
        x = conv(x,256,scope='conv3')
        x = batch_lrelu(x)
        x = attention(x,x.shape[-1])
        print(x)
        x = conv(x,512,scope='conv4')
        x = batch_lrelu(x)
        print(x)
        x = conv(x,1024,scope='conv5')
        x = batch_lrelu(x)
        print(x)
        x = tf.layers.average_pooling2d(x,2,1,'VALID')
        x = fully_conneted(x,1,scope='conv_dense')
        
    return x

def generator(z,trainable=True, reuse=tf.AUTO_REUSE):
        #z = tf.reshape(z,(-1,1,1,n_dim))
        with tf.variable_scope("generator", reuse=reuse):
            z = fully_conneted(z,2*2*1024,scope='deconv_dense')
            z = batch_relu(z)
            z = tf.reshape(z,(-1,2,2,1024))
            z = deconv(z,512,scope='deconv1')
            z = batch_relu(z)
            print(z)
            z = deconv(z,256,scope='deconv2')
            z = batch_relu(z)
            print(z)
            z = deconv(z,128,3,scope='deconv3')
            z = batch_relu(z)
            z = attention(z,z.shape[-1])
            print(z)
            z = deconv(z,64,3,scope='deconv4')
            z = batch_relu(z)
            print(z)
            '''
            z = tf.layers.conv2d_transpose(z,64,3,2,padding='SAME')
            z = batch_relu(z)
            print(z)
            '''
            z = deconv(z,3,scope='deconv5')
            z = tf.nn.tanh(z)
            print(z)
        return z


def conv(x, channels, kernel=5, stride=2,padding='SAME',scope='conv_0'):
    w_init = tf.truncated_normal_initializer()
    with tf.variable_scope(scope):
        w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=w_init)
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                         strides=[1, stride, stride, 1], padding=padding)

        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)
            
    return x


def deconv(x,channels,kernel=3,stride=2,padding='SAME',scope='deconv_0'):
    w_init = tf.truncated_normal_initializer()
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=w_init)
        x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)
    return x

def fully_conneted(x, units,scope='fully_0'):
    w_init = tf.truncated_normal_initializer()
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]
        w = tf.get_variable("kernel", [channels, units], tf.float32,initializer=w_init)
        bias = tf.get_variable("bias", [units],initializer=tf.constant_initializer(0.0))
        x = tf.matmul(x, spectral_norm(w)) + bias
        
    return x

def spectral_norm(w,iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)



def batch_lrelu(x,trainable=True):
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
        x = gamma*o+x
        
    return x

def model_loss(Gz,Dr_logits,Df_logits):
    #if(gan_type=='w_gan'):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - Dr_logits))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + Df_logits))
    D_loss = real_loss+fake_loss
    
    G_loss = -tf.reduce_mean(Df_logits)
    '''
    else:
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dr_logits, labels=tf.ones_like(Dr_logits)))
        
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Df_logits, labels=tf.zeros_like(Df_logits)))
        D_loss = D_loss_real + D_loss_fake
        
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Df_logits, labels=tf.ones_like(Df_logits)))
    '''
    return D_loss,G_loss
'''
def gradient_penalty(real, fake):
    if gan_type == 'drgan' :
        shape = tf.shape(real)
        eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
        noise = 0.5 * x_std * eps  # delta in paper

        # Author suggested U[0,1] in original paper, but he admitted it is bug in github
        # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

        alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
        interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

    else :
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = alpha*real + (1. - alpha)*fake

    logit = discriminator(interpolated, reuse=True)

    grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
    grad = tf.layers.flatten(grad)
    grad_norm = tf.norm(grad, axis=1)  # l2 norm

    GP = 0
    
    # WGAN - LP
    if gan_type == 'sagan':
        GP = ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
    else:
        GP = ld * tf.reduce_mean(tf.square(grad_norm - 1.))

    return GP
'''
def optimizer(D_loss,G_loss):
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_opt = tf.train.AdamOptimizer(lr*2,beta1=0.5).minimize(D_loss, var_list=D_vars)
    
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
    #print(img)
    return img

def get_img(sess,batch_test_x):
    img = sess.run(batch_test_x)
    img = norm(img)
    #print(img[0][0][0])
    return img

if(__name__=='__main__'):
    check_enviroment()
    train_image,train_label = read_and_decode(tfrecord_train,batch_size)
    noise_input = tf.placeholder(tf.float32,[batch_size,n_dim])
    real_img = tf.placeholder(tf.float32,[batch_size,img_size,img_size,channel])
    
    Gz = generator(noise_input)
    Dr_logits = discriminator(real_img)
    Df_logits = discriminator(Gz)
    
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
            #saver.restore(sess, './checkpoint1_dir/MyModel-3400')
        else:
            assert 'can not find checkpoint folder path!'
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        num = 0
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
