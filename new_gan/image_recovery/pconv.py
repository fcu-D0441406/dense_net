import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import cv2
import numpy as np

ckpts = './checkpoint1_dir'
adam_meta = './checkpoint1_dir/MyModel'

def upsample(x):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    return x

def batch_relu(x,trainable=True):
    x = tf.layers.batch_normalization(x,training=trainable)
    return tf.nn.relu(x)
def batch_leaky_relu(x,trainable=True):
    x = tf.layers.batch_normalization(x,training=trainable)
    return tf.nn.leaky_relu(x)

def build_recovery(img,mask):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        img_conv1,mask_conv1 = utils.p_conv2d([img,mask],64,7,2,'SAME','pconv1')#128
        img_conv1 = batch_relu(img_conv1)
        #print(img_conv1,mask_conv1)
        
        img_conv2,mask_conv2 = utils.p_conv2d([img_conv1,mask_conv1],128,5,2,'SAME','pconv2')#64
        img_conv2 = batch_relu(img_conv2)
        #print(img_conv2,mask_conv2)
        
        img_conv3,mask_conv3 = utils.p_conv2d([img_conv2,mask_conv2],256,5,2,'SAME','pconv3')#32
        img_conv3 = batch_relu(img_conv3)
        #print(img_conv3,mask_conv3)
        
        img_conv4,mask_conv4 = utils.p_conv2d([img_conv3,mask_conv3],512,3,2,'SAME','pconv4')#16
        img_conv4 = batch_relu(img_conv4)
        #print(img_conv4,mask_conv4)
        
        img_conv5,mask_conv5 = utils.p_conv2d([img_conv4,mask_conv4],512,3,2,'SAME','pconv5')#8
        img_conv5 = batch_relu(img_conv5)
        #print(img_conv5,mask_conv5)
        
        img_conv6,mask_conv6 = utils.p_conv2d([img_conv5,mask_conv5],512,3,2,'SAME','pconv6')#4
        img_conv6 = batch_relu(img_conv6)
        #print(img_conv6,mask_conv6)
        
        img_conv7,mask_conv7 = utils.p_conv2d([img_conv6,mask_conv6],512,3,2,'SAME','pconv7')#2
        img_conv7 = batch_relu(img_conv7)
        print(img_conv7,mask_conv7)
          
        img_up_conv1,mask_up_conv1 = upsample(img_conv7),upsample(mask_conv7)
        img_up_conv1,mask_up_conv1 = tf.concat([img_conv6,img_up_conv1],axis=-1),tf.concat([mask_conv6,mask_up_conv1],axis=-1)
        img_up_conv1,mask_up_conv1 = utils.p_conv2d([img_up_conv1,mask_up_conv1],512,3,1,'SAME','up_pconv1')#4
        img_up_conv1 = batch_leaky_relu(img_up_conv1)
        
        img_up_conv2,mask_up_conv2 = upsample(img_up_conv1),upsample(mask_up_conv1)
        img_up_conv2,mask_up_conv2 = tf.concat([img_conv5,img_up_conv2],axis=-1),tf.concat([mask_conv5,mask_up_conv2],axis=-1)
        img_up_conv2,mask_up_conv2 = utils.p_conv2d([img_up_conv2,mask_up_conv2],512,3,1,'SAME','up_pconv2')#8
        img_up_conv2 = batch_leaky_relu(img_up_conv2)
        
        img_up_conv3,mask_up_conv3 = upsample(img_up_conv2),upsample(mask_up_conv2)
        img_up_conv3,mask_up_conv3 = tf.concat([img_conv4,img_up_conv3],axis=-1),tf.concat([mask_conv4,mask_up_conv3],axis=-1)
        img_up_conv3,mask_up_conv3 = utils.p_conv2d([img_up_conv3,mask_up_conv3],512,3,1,'SAME','up_pconv3')#16
        img_up_conv3 = batch_leaky_relu(img_up_conv3)
        
        img_up_conv4,mask_up_conv4 = upsample(img_up_conv3),upsample(mask_up_conv3)
        img_up_conv4,mask_up_conv4 = tf.concat([img_conv3,img_up_conv4],axis=-1),tf.concat([mask_conv3,mask_up_conv4],axis=-1)
        img_up_conv4,mask_up_conv4 = utils.p_conv2d([img_up_conv4,mask_up_conv4],256,3,1,'SAME','up_pconv4')#32
        img_up_conv4 = batch_leaky_relu(img_up_conv4)
        
        img_up_conv5,mask_up_conv5 = upsample(img_up_conv4),upsample(mask_up_conv4)
        img_up_conv5,mask_up_conv5 = tf.concat([img_conv2,img_up_conv5],axis=-1),tf.concat([mask_conv2,mask_up_conv5],axis=-1)
        img_up_conv5,mask_up_conv5 = utils.p_conv2d([img_up_conv5,mask_up_conv5],128,3,1,'SAME','up_pconv5')#64
        img_up_conv5 = batch_leaky_relu(img_up_conv5)
        
        img_up_conv6,mask_up_conv6 = upsample(img_up_conv5),upsample(mask_up_conv5)
        img_up_conv6,mask_up_conv6 = tf.concat([img_conv1,img_up_conv6],axis=-1),tf.concat([mask_conv1,mask_up_conv6],axis=-1)
        img_up_conv6,mask_up_conv6 = utils.p_conv2d([img_up_conv6,mask_up_conv6],64,3,1,'SAME','up_pconv6')#128
        img_up_conv6 = batch_leaky_relu(img_up_conv6)
        
        img_recovery,mask_comp = upsample(img_up_conv6),upsample(mask_up_conv6)
        img_recovery,mask_comp = tf.concat([img,img_recovery],axis=-1),tf.concat([mask,mask_comp],axis=-1)
        img_recovery,mask_comp = utils.p_conv2d([img_recovery,mask_comp],3,3,1,'SAME','up_pconv7')#128
        img_recovery = tf.nn.tanh(img_recovery)
        print(img_recovery,mask_comp)
        return img_recovery,mask_comp

def small_build_recovery(img,mask):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        img_conv1,mask_conv1 = utils.p_conv2d([img,mask],32,7,2,'SAME','pconv1')#128
        img_conv1 = batch_relu(img_conv1)
        #print(img_conv1,mask_conv1)
        
        img_conv2,mask_conv2 = utils.p_conv2d([img_conv1,mask_conv1],64,3,2,'SAME','pconv2')#64
        img_conv2 = batch_relu(img_conv2)
        #print(img_conv2,mask_conv2)
        
        img_conv3,mask_conv3 = utils.p_conv2d([img_conv2,mask_conv2],128,3,2,'SAME','pconv3')#32
        img_conv3 = batch_relu(img_conv3)
        #print(img_conv3,mask_conv3)
        
        img_conv4,mask_conv4 = utils.p_conv2d([img_conv3,mask_conv3],256,3,2,'SAME','pconv4')#16
        img_conv4 = batch_relu(img_conv4)
        #print(img_conv4,mask_conv4)
        
        #img_conv5,mask_conv5 = utils.p_conv2d([img_conv4,mask_conv4],512,3,2,'SAME','pconv5')#8
        #img_conv5 = batch_relu(img_conv5)
        #print(img_conv5,mask_conv5)
        
        #img_conv6,mask_conv6 = utils.p_conv2d([img_conv5,mask_conv5],512,3,2,'SAME','pconv6')#4
        #img_conv6 = batch_relu(img_conv6)
        #print(img_conv6,mask_conv6)
        
        #img_conv7,mask_conv7 = utils.p_conv2d([img_conv6,mask_conv6],512,3,2,'SAME','pconv7')#2
        #img_conv7 = batch_relu(img_conv7)
        #print(img_conv7,mask_conv7)
          
        #img_up_conv1,mask_up_conv1 = upsample(img_conv7),upsample(mask_conv7)
        #img_up_conv1,mask_up_conv1 = tf.concat([img_conv6,img_up_conv1],axis=-1),tf.concat([mask_conv6,mask_up_conv1],axis=-1)
        #img_up_conv1,mask_up_conv1 = utils.p_conv2d([img_up_conv1,mask_up_conv1],512,3,1,'SAME','up_pconv1')#4
        #img_up_conv1 = batch_leaky_relu(img_up_conv1)
        
        #img_up_conv2,mask_up_conv2 = upsample(img_conv6),upsample(mask_conv6)
        #img_up_conv2,mask_up_conv2 = tf.concat([img_conv5,img_up_conv2],axis=-1),tf.concat([mask_conv5,mask_up_conv2],axis=-1)
        #img_up_conv2,mask_up_conv2 = utils.p_conv2d([img_up_conv2,mask_up_conv2],512,3,1,'SAME','up_pconv2')#8
        #img_up_conv2 = batch_leaky_relu(img_up_conv2)
        
        #img_up_conv3,mask_up_conv3 = upsample(img_conv5),upsample(mask_conv5)
        #img_up_conv3,mask_up_conv3 = tf.concat([img_conv4,img_up_conv3],axis=-1),tf.concat([mask_conv4,mask_up_conv3],axis=-1)
        #img_up_conv3,mask_up_conv3 = utils.p_conv2d([img_up_conv3,mask_up_conv3],512,3,1,'SAME','up_pconv3')#16
        #img_up_conv3 = batch_leaky_relu(img_up_conv3)
        
        img_up_conv4,mask_up_conv4 = upsample(img_conv4),upsample(mask_conv4)
        img_up_conv4,mask_up_conv4 = tf.concat([img_conv3,img_up_conv4],axis=-1),tf.concat([mask_conv3,mask_up_conv4],axis=-1)
        img_up_conv4,mask_up_conv4 = utils.p_conv2d([img_up_conv4,mask_up_conv4],128,3,1,'SAME','up_pconv4')#32
        img_up_conv4 = batch_leaky_relu(img_up_conv4)
        
        img_up_conv5,mask_up_conv5 = upsample(img_up_conv4),upsample(mask_up_conv4)
        img_up_conv5,mask_up_conv5 = tf.concat([img_conv2,img_up_conv5],axis=-1),tf.concat([mask_conv2,mask_up_conv5],axis=-1)
        img_up_conv5,mask_up_conv5 = utils.p_conv2d([img_up_conv5,mask_up_conv5],64,3,1,'SAME','up_pconv5')#64
        img_up_conv5 = batch_leaky_relu(img_up_conv5)
        
        img_up_conv6,mask_up_conv6 = upsample(img_up_conv5),upsample(mask_up_conv5)
        img_up_conv6,mask_up_conv6 = tf.concat([img_conv1,img_up_conv6],axis=-1),tf.concat([mask_conv1,mask_up_conv6],axis=-1)
        img_up_conv6,mask_up_conv6 = utils.p_conv2d([img_up_conv6,mask_up_conv6],32,3,1,'SAME','up_pconv6')#128
        img_up_conv6 = batch_leaky_relu(img_up_conv6)
        
        img_recovery,mask_comp = upsample(img_up_conv6),upsample(mask_up_conv6)
        img_recovery,mask_comp = tf.concat([img,img_recovery],axis=-1),tf.concat([mask,mask_comp],axis=-1)
        img_recovery,mask_comp = utils.p_conv2d([img_recovery,mask_comp],3,3,1,'SAME','up_pconv7')#256
        img_recovery = tf.nn.tanh(img_recovery)
        print(img_recovery,mask_comp)
        return img_recovery,mask_comp
'''
def build_gate_conv(img,mask):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
        x = tf.concat([img,mask],axis=-1)
        img_conv1 = utils.gate_conv(x,32,7,2,name='conv',activation='relu')
        img_conv2 = utils.gate_conv(img_conv1,64,3,2,name='conv1',activation='relu')
        img_conv3 = utils.gate_conv(img_conv2,128,3,2,name='conv2',activation='relu')
        img_conv4 = utils.gate_conv(img_conv3,256,3,2,name='conv3',activation='relu')
        img_conv5 = utils.gate_conv(img_conv4,512,3,2,name='conv4',activation='relu')
        
        img_up_conv = 
'''   
        

def discriminator(image,reuse=tf.AUTO_REUSE):
    with tf.variable_scope('discriminator',reuse=reuse):
        conv1 = utils.conv(image,64,kernel=7,stride=2, pad=3,scope='d_conv1')
        conv1 = batch_leaky_relu(conv1)
        print(conv1)
        conv2 = utils.conv(conv1,128,kernel=5,stride=2, pad=2,scope='d_conv2')
        conv2 = batch_leaky_relu(conv2)
        print(conv2)
        conv3 = utils.conv(conv2,256,kernel=5,stride=2, pad=2,scope='d_conv3')
        conv3 = batch_leaky_relu(conv3)
        print(conv3)
        
        conv4 = utils.conv(conv3,512,kernel=5,stride=2, pad=2,scope='d_conv4')
        conv4 = batch_leaky_relu(conv4)
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
        D_opt = tf.train.AdamOptimizer(1e-4*4).minimize(D_loss, var_list=D_vars)
    
        G_opt = tf.train.AdamOptimizer(1e-4).minimize(G_loss, var_list=G_vars)
    return D_opt,G_opt

def show_img(img):
    plt.imshow(img)
    plt.show()

def gen_trainsform(img):
    img = (img+1)/2
    return img

def img_norm(img):
    img = (img/127.5)-1
    return img

original_img = tf.placeholder(tf.float32,[None,256,256,3])
noise_img = tf.placeholder(tf.float32,[None,256,256,3])
mask = tf.placeholder(tf.float32,[None,256,256,3])

recovery_img,mask_comp = small_build_recovery(noise_img,mask)
'''
Dr_logits = discriminator(original_img)
Df_logits = discriminator(recovery_img)
'''
Dr_logits = discriminator(tf.concat([noise_img,original_img],axis=-1))
Df_logits = discriminator(tf.concat([noise_img,recovery_img],axis=-1))


D_loss,G_loss = model_loss(Dr_logits,Df_logits)
pixel_loss = utils.pixel_loss(mask,original_img,recovery_img)
G_loss = G_loss+pixel_loss
#cycle_loss = tf.reduce_mean(tf.abs(original_img-recovery_img))
#G_loss = G_loss+5*cycle_loss
D_opt,G_opt = optimizer(D_loss,G_loss)


inputs_img = cv2.cvtColor(cv2.imread('./train18.jpg'),cv2.COLOR_BGR2RGB)
inputs_mask = np.ones((256,256,3))
inputs_mask_img = inputs_img.copy()
#show_img(inputs_img)
inputs_mask_img[60:140,100:180,:] = [255,255,255]
show_img(inputs_mask_img)
inputs_mask[60:140,100:180,:] = [0,0,0]

inputs_img = img_norm(inputs_img[np.newaxis,:,:,:])
inputs_mask_img = img_norm(inputs_mask_img[np.newaxis,:,:,:])

inputs_img_ = np.zeros((4,256,256,3))
inputs_mask_img_ = np.zeros((4,256,256,3))
inputs_mask_ = np.zeros((4,256,256,3))

for i in range(4):
    inputs_img_[i] = inputs_img.copy()
    inputs_mask_img_[i] = inputs_mask_img.copy()
    inputs_mask_[i] = inputs_mask.copy()
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(ckpts) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(ckpts))
    else:
        assert 'can not find checkpoint folder path!'
    
    for i in range(10000):
            
        _, d_loss = sess.run([D_opt, D_loss],
                                  feed_dict={original_img:inputs_img_,noise_img:inputs_mask_img_,mask:inputs_mask_})
        
        _, g_loss = sess.run([G_opt, G_loss],
                                  feed_dict={original_img:inputs_img_,noise_img:inputs_mask_img_,mask:inputs_mask_})
        
        if(i%100==0):
            gz = sess.run(recovery_img,feed_dict={original_img:inputs_img_,noise_img:inputs_mask_img_,mask:inputs_mask_})
            gz = gen_trainsform(gz[0])
            show_img(gz)
            saver.save(sess,adam_meta,global_step=i)
        
        print('step:',i,'d_loss:',d_loss,'g_loss:',g_loss)
