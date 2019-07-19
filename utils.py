import tensorflow as tf
from keras import backend as K

def p_conv2d(inputs,filters,kernel_size,stride,padding='SAME',scope=''):
    with tf.variable_scope(scope):
        
        normalize = tf.reduce_mean(inputs[1],axis=[1,2],keepdims=True)
        normalize = tf.keras.backend.repeat_elements(normalize,inputs[0].shape[1],axis=1)
        normalize = tf.keras.backend.repeat_elements(normalize,inputs[0].shape[2],axis=2)

        image_feature = inputs[0]
        mask_feature = inputs[1]
        print(image_feature,mask_feature)
        image_feature_out = tf.layers.conv2d((image_feature*mask_feature)/normalize,filters,
                                             kernel_size,stride,padding=padding)
        mask_feature_out = tf.layers.conv2d(mask_feature,filters,kernel_size,stride,padding=padding)

        bias = tf.get_variable("bias", [image_feature_out.shape[-1]], initializer=tf.constant_initializer(0.0))
        image_feature_out = tf.nn.bias_add(image_feature_out,bias)

        return [image_feature_out,mask_feature_out]