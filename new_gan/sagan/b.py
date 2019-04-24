import tensorflow as tf
import numpy as np

class BEGAN(object):
    def __init__(self, place_holder=''):
        self.place_holder = place_holder
        # pass

    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs/tensors
        """
        inputs_real = tf.placeholder(
            tf.float32, (None, image_width, image_height, image_channels), name='input_real')
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        k_t = tf.placeholder(tf.float32, name='k_t')

        return inputs_real, inputs_z, learning_rate, k_t


    def fully_connected(self, x, output_shape):
        # flatten and dense
        shape = x.get_shape().as_list()
        dim = np.prod(shape[1:])

        x = tf.reshape(x, [-1, dim])
        x = tf.layers.dense(x, output_shape, activation=None)

        return x

    def decoder(self, h, n, img_dim, channel_dim):
        """
        Reconstruction network
        """
        h = tf.layers.dense(h, img_dim * img_dim * n, activation=None)
        h = tf.reshape(h, (-1, img_dim, img_dim, n))

        conv1 = tf.layers.conv2d(
            h, n, 3, padding="same", activation=tf.nn.leaky_relu)
        conv1 = tf.layers.conv2d(
            conv1, n, 3, padding="same", activation=tf.nn.leaky_relu)

        upsample1 = tf.image.resize_nearest_neighbor(
            conv1, size=(img_dim * 2, img_dim * 2))

        conv2 = tf.layers.conv2d(
            upsample1, n, 3, padding="same", activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(
            conv2, n, 3, padding="same", activation=tf.nn.leaky_relu)

        upsample2 = tf.image.resize_nearest_neighbor(
            conv2, size=(img_dim * 4, img_dim * 4))

        conv3 = tf.layers.conv2d(
            upsample2, n, 3, padding="same", activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(
            conv3, n, 3, padding="same", activation=tf.nn.leaky_relu)

        conv4 = tf.layers.conv2d(conv3, channel_dim, 3,
                                 padding="same", activation=None)

        return conv4

    def encoder(self, images, n, z_dim, channel_dim):
        """
        Feature extraction network
        """
        conv1 = tf.layers.conv2d(
            images, n, 3, padding="same", activation=tf.nn.leaky_relu)

        conv2 = tf.layers.conv2d(
            conv1, n, 3, padding="same", activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(
            conv2, n * 2, 3, padding="same", activation=tf.nn.leaky_relu)

        subsample1 = tf.layers.conv2d(
            conv2, n * 2, 3, strides=2, padding='same')

        conv3 = tf.layers.conv2d(subsample1, n * 2, 3,
                                 padding="same", activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(
            conv3, n * 3, 3, padding="same", activation=tf.nn.leaky_relu)

        subsample2 = tf.layers.conv2d(
            conv3, n * 3, 3, strides=2, padding='same')

        conv4 = tf.layers.conv2d(subsample2, n * 3, 3,
                                 padding="same", activation=tf.nn.leaky_relu)
        conv4 = tf.layers.conv2d(
            conv4, n * 3, 3, padding="same", activation=tf.nn.leaky_relu)

        h = self.fully_connected(conv4, z_dim)

        return h

    def discriminator(self, images, z_dim, channel_dim, reuse=False):
        """
        Create the discriminator network: The autoencoder
        先encoder成噪音 在decorder成圖像
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            x = self.encoder(images, 64, z_dim, channel_dim)
            x = self.decoder(x, 64, 64 // 4, channel_dim)

            return x

    def generator(self, z, channel_dim, is_train=True):
        """
        Create the generator network: Only the encoder part
        """
        reuse = False if is_train else True
        with tf.variable_scope('generator', reuse=reuse):
            x = self.decoder(z, 64, 64 // 4, channel_dim)

            return x

    def model_loss(self, input_real, input_z, channel_dim, z_dim, k_t):
        """
        Get the loss for the discriminator and generator
        """
        g_model_fake = self.generator(input_z, channel_dim, is_train=True)
        d_model_real = self.discriminator(input_real, z_dim, channel_dim)
        d_model_fake = self.discriminator(
            g_model_fake, z_dim, channel_dim, reuse=True)

        # l1 loss
        d_real = tf.reduce_mean(tf.abs(input_real - d_model_real))
        d_fake = tf.reduce_mean(tf.abs(g_model_fake - d_model_fake))

        d_loss = d_real - k_t * d_fake
        g_loss = d_fake

        return d_loss, g_loss, d_real, d_fake

    def model_opt(self, d_loss, g_loss, learning_rate, beta1, beta2=0.999):
        """
        Get optimization operations
        """
        # Get variables
        g_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, "generator")
        d_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")

        # Optimize
        d_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

