import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import resnet_v2
import os
import dense_net
from matplotlib import pyplot

ckpts = './checkpoint2_dir'
adam_meta = './checkpoint2_dir/MyModel'
tfrecord_train = './train.tfrecords'
tfrecord_test = './test.tfrecords'
batch_size = 64
CLASS_NUM = 2
step = int(23500/batch_size)
epoch = 50
img_size = 64
test_size = 64
channel=3

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

def label_convert_onehot(labels):
    n_sample = len(labels)
    n_class = CLASS_NUM
    one_hot_labels = np.zeros((n_sample,n_class))
    one_hot_labels[np.arange(n_sample),labels] = 1
    return np.array(one_hot_labels,dtype = np.uint8)

if(__name__=='__main__'):
    
    train_image,train_label = read_and_decode(tfrecord_train,batch_size)
    test_image,test_label = read_and_decode(tfrecord_test,test_size)
    print(train_image,test_image)
    #resnet32 = resnet_v2.ResNet(CLASS_NUM)
    resnet32 = dense_net.Dense_net(img_size,channel,CLASS_NUM,16,0.5)
    #resnet32 = resnet_v2.ResNet_50(CLASS_NUM)
    x = resnet32.x
    y = tf.placeholder(tf.float32,[None,CLASS_NUM])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), 
                                                                             logits=resnet32.prediction))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(resnet32.prediction,1)),'float'))
        tf.summary.scalar('ACC', accuracy)
    ########## batch_nor 方法
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=5000,
                                               decay_rate=0.1, staircase=True)
    #opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')
    with tf.name_scope('opt'):
        opt = tf.train.AdamOptimizer(1e-3,name='optimizer')
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(loss)
            train_op = opt.apply_gradients(grads, global_step=global_step)
        #tf.summary.scalar('opti', opt)
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('C:/logfile', sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        
        ######## batch_norm save
        var_list = tf.trainable_variables()
        if global_step is not None:
            var_list.append(global_step)
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list,max_to_keep=5)
        ########
        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        print('start')
        init_step = 0
        for j in range(epoch):
            for i in range(step):
                batch_x,batch_y = sess.run([train_image,train_label])
                batch_y = label_convert_onehot(batch_y)
                _,step = sess.run([train_op, global_step],feed_dict={x:batch_x,y:batch_y})
                init_step+=1
                if(i%100==0):
                    batch_test_x,batch_test_y = sess.run([test_image,test_label])
                    batch_test_y = label_convert_onehot(batch_test_y)
                    ls,acc = sess.run([loss,accuracy],feed_dict={x:batch_x,y:batch_y})
                    val_ls,val_acc = sess.run([loss,accuracy],feed_dict={x:batch_test_x,y:batch_test_y})
                    pre = sess.run(resnet32.prediction,feed_dict={x:batch_test_x})
                    result = sess.run(merged,feed_dict={x:batch_x,y:batch_y})
                    writer.add_summary(result, step)
                    print('epoch ',j,' ls ',ls,' acc ',acc,' val_ls ',val_ls,' val_acc ',val_acc)
            
            saver.save(sess,adam_meta,global_step=step) 
        
        coord.request_stop()
        coord.join(threads)