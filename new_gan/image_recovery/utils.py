import tensorflow as tf
import tensorflow.contrib as tf_contrib
from keras import backend as K

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)


        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                     initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        return x + x_init

##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def spectral_norm(w, iteration=1):
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


def p_conv2d(inputs,filters,kernel_size,stride,padding='SAME',scope=''):
    with tf.variable_scope(scope):
        
        normalize = tf.reduce_mean(inputs[1],axis=[1,2],keepdims=True)
        normalize = tf.keras.backend.repeat_elements(normalize,inputs[0].shape[1],axis=1)
        normalize = tf.keras.backend.repeat_elements(normalize,inputs[0].shape[2],axis=2)

        image_feature = inputs[0]
        mask_feature = inputs[1]
        
        image_feature_out = tf.layers.conv2d((image_feature*mask_feature)/normalize,filters,
                                             kernel_size,stride,padding=padding)
        mask_feature_out = tf.layers.conv2d(mask_feature,filters,kernel_size,stride,padding=padding)

        bias = tf.get_variable("bias", [image_feature_out.shape[-1]], initializer=tf.constant_initializer(0.0))
        image_feature_out = tf.nn.bias_add(image_feature_out,bias)

        return [image_feature_out,mask_feature_out]
'''
def p_conv2d(inputs,filters,kernel_size,stride,padding='SAME',scope=''):
    with tf.variable_scope(scope):
        
        normalize = tf.reduce_mean(inputs[1],axis=[1,2],keepdims=True)
        normalize = tf.keras.backend.repeat_elements(normalize,inputs[0].shape[1],axis=1)
        normalize = tf.keras.backend.repeat_elements(normalize,inputs[0].shape[2],axis=2)

        image_feature = inputs[0]
        mask_feature = inputs[1]
        
        image_feature_out = conv((image_feature*mask_feature)/normalize,filters,
                                             kernel_size,stride,pad=kernel_size//2,scope='sn_conv1')
        mask_feature_out = conv(mask_feature,filters,kernel_size,stride,pad=kernel_size//2,scope='sn_conv2')

        bias = tf.get_variable("bias", [image_feature_out.shape[-1]], initializer=tf.constant_initializer(0.0))
        image_feature_out = tf.nn.bias_add(image_feature_out,bias)

        return [image_feature_out,mask_feature_out]
'''
def get_total_loss(original_image,recovery_image,original_mask,recovery_mask):
    pass

def pixel_loss(original_mask,original_image,recovery_image):
    mask_loss = tf.reduce_mean(tf.abs((1-original_mask)*original_image - (1-original_mask)*recovery_image))
    not_mask_loss = tf.reduce_mean(tf.abs(original_mask*original_image - original_mask*recovery_image))
    return tf.add(mask_loss,3*not_mask_loss)

def gate_conv(x_in, cnum, ksize, stride=1, rate=1, name='conv',
              padding='SAME', activation='leaky_relu', use_lrn=True, training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x_in, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if use_lrn:
        x = tf.nn.lrn(x, bias=0.00005)
    if activation == 'leaky_relu':
        x = tf.nn.leaky_relu(x)
    if activation == 'relu':
        x = tf.nn.relu(x)

    g = tf.layers.conv2d(
        x_in, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name + '_g')

    x = tf.multiply(x, g)
    return x, g