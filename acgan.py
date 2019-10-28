import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import ops_v2
from tensorflow.examples.tutorials.mnist import input_data

ld = 10.0
img_size = 32
CLASS_NUM = 10
channel = 1
batch_size = 64
per_img = 8
n_dim = 64
lr = 0.00005
epoch = 10000
gan_type = 'normal'

ckpts = './checkpoint3_dir'
adam_meta = './checkpoint3_dir/MyModel'
output_dir = './output3'
#ck3 output3
#ck4 output4
mnist = input_data.read_data_sets('./MNIST', one_hot=True)

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
        #ch = 32
        '''
        x = ops_v2.snconv2d(x,32,5,5,1,1,name='init_block')
        #x = tf.nn.leaky_relu(x)
        
        x = risidual_down_block(x,ch*2,'conv0')#14
        print(x)
        
        x = risidual_down_block(x,ch*4,'conv1')#7
        x = attention(x,x.shape[-1])
        print(x)
        
        x = risidual_down_block(x,ch*8,'conv2')#4
        print(x)
        
        x = risidual_down_block(x,ch*8,'conv3') #16*16*512
        print(x)
        
        x = risidual_down_block(x,ch*16,'conv4') #8*8*1024
        print(x)
        
        x = risidual_down_block(x,ch*32,'conv5') #4*4*1024
        print(x)
        
        x = tf.reduce_sum(x, [1, 2])
        print(x)
        predict_label = ops_v2.snlinear(x,10,name='label_liner')
        x = ops_v2.snlinear(x,1,name='d_sn_linear')
        print(x,predict_label)
        '''
        x = tf.layers.conv2d(x,64,5,2,padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x)
        
        x = tf.layers.conv2d(x,128,3,2,padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x)
        x = attention(x,x.shape[-1])
        
        x = tf.layers.conv2d(x,256,3,2,padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x)
        
        x = tf.layers.flatten(x)
        predict_label = tf.layers.dense(x,10)
        x = tf.layers.dense(x,1)
    return x,predict_label

def generator(z,real_label,trainable=True, reuse=tf.AUTO_REUSE):
    #z = tf.reshape(z,(-1,1,1,n_dim))
    with tf.variable_scope("generator", reuse=reuse):
        ch = 256
        z = tf.concat([z,real_label],axis=-1)
        z2 = tf.layers.dense(z,ch*4*4)
        z2 = tf.layers.batch_normalization(z2)
        z2 = tf.nn.relu(z2)
        z2 = tf.reshape(z2,(-1,4,4,ch))
        
        z2 = tf.layers.conv2d_transpose(z2,128,3,2,padding='SAME')
        z2 = tf.layers.batch_normalization(z2)
        z2 = tf.nn.relu(z2)
        z2 = attention(z2,z2.shape[-1])
        
        z2 = tf.layers.conv2d_transpose(z2,64,3,2,padding='SAME')
        z2 = tf.layers.batch_normalization(z2)
        z2 = tf.nn.relu(z2)
        
        z2 = tf.layers.conv2d_transpose(z2,1,5,2,padding='SAME')
        
        '''
        z2 = ops_v2.linear(z,ch*4*4,scope='g_h0')
        z2 = tf.reshape(z2,(-1,4,4,ch))#4*4*1024
        print(z2)
        
        z2 = risidual_up_block(z2,z,ch//2,trainable,scope='deconv0')#8*8*1024
        print(z2)
        
        z2 = risidual_up_block(z2,z,ch//4,trainable,scope='deconv1')#16*16*512
        z2 = attention(z2,z2.shape[-1])
        print(z2)
        
        z2 = risidual_up_block(z2,z,ch//8,trainable,scope='deconv2')#32*32*256
        print(z2)
        
        z2 = risidual_up_block(z2,ch//8,trainable,scope='deconv3')#64*64*128
        print(z)
        
        z = risidual_up_block(z,ch//16,trainable,scope='deconv4')#128*128*64
        print(z)
        
        z = risidual_up_block(z,ch//32,trainable,scope='deconv5')#256*256*64
        print(z)
        
        z2 = tf.layers.batch_normalization(z2,training=trainable)
        z2 = tf.nn.relu(z2)
        z2 = ops_v2.snconv2d(z2,channel,3,3,1,1,name='last_layer')
        '''
        z2 = tf.nn.tanh(z2)
        print(z2)
    return z2

def upsample(x):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    return x

def risidual_up_block(x,z,ch,trainable,scope=''):
    with tf.variable_scope(scope):
        short_cut = x
        
        x = tf.layers.batch_normalization(x,training=trainable)
        #x = condition_batch_norm(x,z,scope='batch0')
        x = tf.nn.relu(x)
        x = ops_v2.snconv2d(x,ch//4,1,1,1,1,name='sn_upconv0')
        #x = condition_batch_norm(x,z,scope='batch1')
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = upsample(x)
        x = ops_v2.snconv2d(x,ch,3,3,1,1,name='sn_upconv1')
        #x = condition_batch_norm(x,z,scope='batch2')
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = ops_v2.snconv2d(x,ch,3,3,1,1,name='sn_upconv2')
        #x = condition_batch_norm(x,z,scope='batch3')
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = ops_v2.snconv2d(x,ch,1,1,1,1,name='sn_upconv3')
        
        short_cut = upsample(short_cut)
        short_cut = ops_v2.snconv2d(x,ch,3,3,1,1,name='sn_sh_upconv')
    return x+short_cut

def risidual_down_block(x,ch,scope):
    with tf.variable_scope(scope):
        short_cut = x
        x = tf.nn.relu(x)
        x = ops_v2.snconv2d(x,ch//4,1,1,1,1,name='sn_conv0')
        x = tf.nn.relu(x)
        x = ops_v2.snconv2d(x,ch, 3, 3, 1, 1,name='sn_conv1')
        x = tf.nn.relu(x)
        x = ops_v2.snconv2d(x,ch,3,3,1,1,name='sn_conv2')
        x = tf.nn.relu(x)
        x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        x = ops_v2.snconv2d(x,ch,1,1,1,1,name='sn_conv3')
        
        short_cut = tf.nn.avg_pool(short_cut, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        short_cut2 = ops_v2.snconv2d(short_cut,short_cut.shape[-1],1,1,1,1,name='sc_sn_conv')
        short_cut = Concatenation([short_cut,short_cut2])
        
    return x+short_cut

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def attention(x, ch, scope='attention', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        f = ops_v2.snconv2d(x,ch//8,1,1,1,1,name='f_conv')
        g = ops_v2.snconv2d(x,ch//8,1,1,1,1,name='g_conv')
        h = ops_v2.snconv2d(x,ch,1,1,1,1,name='h_conv')
        
        f = tf.reshape(f,(batch_size,-1,f.shape[-1]))
        g = tf.reshape(g,(batch_size,-1,g.shape[-1]))
        h = tf.reshape(h,(batch_size,-1,h.shape[-1]))
        
        s = tf.matmul(g,f,transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.matmul(beta,h)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o,shape=x.shape)
        o = ops_v2.snconv2d(o, ch,1,1,1,1,name='attn_conv')
        x = gamma*o+x
        
    return x

def condition_batch_norm(x, z, is_training=True, scope='batch_norm'):
    with tf.variable_scope(scope) :
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

        beta = ops_v2.snlinear(z, c, name='beta')
        gamma = ops_v2.snlinear(z, c, name='gamma')

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

def model_loss(Gz,Dr_logits,Df_logits,Dr_predict,Df_predict,real_label):
    if(gan_type=='sagan'):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - Dr_logits))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + Df_logits))
        
        Dr_cls_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_label, 1), 
                                                                             logits=Dr_predict))
        Df_cls_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_label, 1), 
                                                                             logits=Df_predict))
        
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
    
    Dr_cls_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_label, 1), 
                                                                             logits=Dr_predict))
    Df_cls_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_label, 1), 
                                                                         logits=Df_predict))
    D_loss+=Dr_cls_loss
    G_loss+=Df_cls_loss
    
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
    if not os.path.exists('output1/'):
        os.makedirs('output1/')
    if not os.path.exists(ckpts):
        os.makedirs(ckpts)

def noise_test_sample(batch_size,noise_dim):
    label = np.zeros([batch_size,CLASS_NUM])
    for i in range(batch_size):
        label[i][i%10] = 1
    return np.random.uniform(-1., 1., size=[batch_size, noise_dim]),label

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

def cv_write_img(G_sample):
    k = 0
    #print(G_sample)
    for img in G_sample:
        img = img*255.0
        img = np.array(img,dtype=np.int32)
        save_img = './output2/'+str(k)+'.jpg'
        cv2.imwrite(save_img,img)
        k+=1

if(__name__=='__main__'):
    
    check_enviroment()
    with tf.Graph().as_default():
        #image_list = get_data('./img2')
        #image_idx = np.arange(len(image_list))
        #os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2'
        noise_input = tf.placeholder(tf.float32,[batch_size,n_dim])
        #noise_input = tf.truncated_normal(shape=[batch_size,n_dim], name='random_z')
        real_img = tf.placeholder(tf.float32,[batch_size,img_size,img_size,channel])
        real_label = tf.placeholder(tf.float32,[None,CLASS_NUM])
        Dr_logits,Dr_predict = discriminator(real_img)
        Gz = generator(noise_input,real_label)
        Df_logits,Df_predict = discriminator(Gz,reuse=True)

        D_loss,G_loss = model_loss(Gz,Dr_logits,Df_logits,Dr_predict,Df_predict,real_label)

        D_opt,G_opt = optimizer(D_loss,G_loss)
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if tf.train.latest_checkpoint(ckpts) is not None:
                saver.restore(sess, tf.train.latest_checkpoint(ckpts))
            else:
                assert 'can not find checkpoint folder path!'

            num = 0
            for i in range(epoch):
                #np.random.shuffle(image_idx)
                for j in range(50000//batch_size):
                    if(j%50==0):
                        g_noise,g_label = noise_test_sample(batch_size,n_dim)
                        G_sample = sess.run(Gz,feed_dict={noise_input:g_noise,real_label:g_label})
                        G_sample = gen_trainsform(G_sample)
                        cv_write_img(G_sample)
                        fig = plot(G_sample)
                        plt.savefig('output3/{}.png'.format(str(num).zfill(3)), bbox_inches='tight')
                        num += 1
                        plt.close(fig)
                        if(num==100):
                            num = 0

                    #input_imgs = get_img(image_list,image_idx[j*batch_size:(j+1)*batch_size])
                    batch_x,input_labels = mnist.train.next_batch(batch_size)
                    batch_x = np.reshape(batch_x,(-1,28,28))
                    input_imgs = np.zeros((batch_size,32,32,1))
                    for i in range(batch_size):
                        input_imgs[i] = np.reshape(cv2.resize(batch_x[i],(32,32)),(32,32,1))
                    input_imgs = (input_imgs*2)-1
                    #print(input_imgs.shape,input_labels.shape,)
                    _, d_loss = sess.run([D_opt, D_loss],
                                              feed_dict={real_img:input_imgs,noise_input:noise_sample(batch_size,n_dim),real_label:input_labels})

                    _, g_loss = sess.run([G_opt, G_loss],
                                              feed_dict={real_img:input_imgs,noise_input: noise_sample(batch_size, n_dim),real_label:input_labels})
                    '''
                    _, d_loss = sess.run([D_opt, D_loss],
                                              feed_dict={real_img:input_imgs})
                    _, g_loss = sess.run([G_opt, G_loss],
                                              feed_dict={real_img:input_imgs})
                    '''
                    print('step:',i,'d_loss:',d_loss,'g_loss:',g_loss)

                saver.save(sess,adam_meta,global_step=i) 