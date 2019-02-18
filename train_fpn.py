import tensorflow as tf
import resnet_v2
import cv2
import numpy as np

img_size = 64
class_num=2
channel = 3
batch_size = 4
test_size = 1
tfrecord_train = './fpn_train.tfrecords'
tfrecord_test = './fpn_test.tfrecords'

def label_convert(result):
    x = np.zeros((img_size,img_size,3))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            r = np.argmax(result[i][j])
            if(r==1):
                x[i][j][0]=255
    
    cv2.imshow('img',x)
    cv2.waitKey()
    cv2.destroyAllWindows()

def read_and_decode(tfrecord_file_path,batch_size):
    tfrecord_file = tf.train.string_input_producer([tfrecord_file_path])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(tfrecord_file)
    img_features = tf.parse_single_example(serialized_example,features={
                                        'label_raw':tf.FixedLenFeature([],tf.string),
                                        'image_raw':tf.FixedLenFeature([],tf.string),
                                        })
    image = tf.decode_raw(img_features['image_raw'],tf.uint8)
    image = tf.reshape(image,[img_size,img_size,channel])
    label = tf.decode_raw(img_features['label_raw'],tf.uint8)
    label = tf.reshape(label,[img_size,img_size,class_num])
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=batch_size,
                                                     min_after_dequeue=5,
                                                     num_threads=4,
                                                     capacity=7)
    return image_batch,label_batch

if(__name__=='__main__'):
    
    train_image,train_label = read_and_decode(tfrecord_train,batch_size)
    test_image,test_label = read_and_decode(tfrecord_test,test_size)
    
    y = tf.placeholder(tf.float32,[None,img_size,img_size,class_num])
    resnet32 = resnet_v2.ResNet(class_num)
    x = resnet32.x
    #resnet32.unsample2()
    fpn_predict = resnet32.unconv5
    
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 3), 
                                                                             logits=fpn_predict))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,3),tf.argmax(fpn_predict,3)),'float'))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(2001):
            batch_x,batch_y = sess.run([train_image,train_label])
            _,ls,acc = sess.run([train_op,loss,accuracy],feed_dict={x:batch_x,y:batch_y})
           
            if(i%100==0):
                
                if(i%200==0and i>=10):
                    batch_test_x,batch_test_y = sess.run([test_image,test_label])
                    result,val_acc = sess.run([fpn_predict,accuracy],feed_dict={x:batch_test_x,y:batch_test_y})
                    result2 = np.reshape(result,(64,64,2))
                    #label_convert(result2)
                    print('val acc',val_acc)
                print(ls,acc)
        coord.request_stop()
        coord.join(threads)
    