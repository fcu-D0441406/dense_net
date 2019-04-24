import cv2
import tensorflow as tf
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import random
import matplotlib.pyplot as plt

tfrecord_train = './train2.tfrecords'
tfrecord_test = './test.tfrecords'
train_file_name = 'train'
test_file_name = 'test'
img_size = 64
channel = 3

def strong_data(img):
    with tf.Session() as sess:
        '''
        img_constrast = tf.image.random_contrast(img,lower=0.5, upper=1.5)
        img_hue = tf.image.random_hue(img, max_delta=0.05)
        ima_brightness = tf.image.random_brightness(img, max_delta=0.2)
        random_brightness = ima_brightness(session=sess)
        random_contrast= img_constrast(session=sess)
        random_hue= img_hue(session=sess)
        '''
        image_h = tf.image.random_hue(img, max_delta=0.05)
        image_b = tf.image.random_brightness(img, max_delta=0.2)
        iamge_c = tf.image.random_contrast(img,lower=0.5, upper=1.5)
        img_h = image_h.eval(session=sess)
        img_b = image_b.eval(session=sess)
        img_c = iamge_c.eval(session=sess)
        return img_h,img_b,img_c
    

def read_data(file_path):
    for root,dirs,files in os.walk(file_path):
        for sub_dir in dirs:
            for root2,_,files2 in os.walk(os.path.join(root,sub_dir)):
                for file in files2:
                    file_path = os.path.join(root2,file)
                    file_path2 = os.path.join(root2,file[:-4]+'b.jpg')
                    file_path3 = os.path.join(root2,file[:-4]+'c.jpg')
                    file_path4 = os.path.join(root2,file[:-4]+'h.jpg')
                    image = cv2.imread(file_path)
                    resize_img = cv2.resize(image,(img_size,img_size))
                    #rb,rc,rh = strong_data(resize_img)
                    rh,rb,rc = strong_data(resize_img)
                    cv2.imwrite(file_path,resize_img)
                    cv2.imwrite(file_path2,rb)
                    cv2.imwrite(file_path3,rc)
                    cv2.imwrite(file_path4,rh)
    
def get_data(file_path):
    image = []
    temp = []
    for root,dirs,files in os.walk(file_path):
        for name in files:
            image.append(os.path.join(root,name))

        for sub_dir in dirs:
            temp.append(os.path.join(root,sub_dir))
        
    labels = []
    for folder in temp:
        n_img = len(os.listdir(folder))
        letter = folder.split('\\')[-1]
        labels = np.append(labels,n_img*[letter])
    
    temp = np.array([image,labels])
    temp = temp.transpose()#維樹反轉 [2,25000] -> [25000,2]
    np.random.shuffle(temp) #numpy的打亂函數
    
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    
    return image_list,label_list

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images_list,labels_list,save_dir,name):
    tfrecord_filename = os.path.join(save_dir,name+'.tfrecords')
    n_samples = len(labels_list)
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    print('\nTransform start')
    for i in np.arange(0,n_samples):
        if(channel==1):
            image = cv2.imread(images_list[i],0)
        elif(channel==3):
            image = cv2.imread(images_list[i])
        image = cv2.resize(image,(img_size,img_size))
        image = np.reshape(image,(img_size,img_size,channel))
        image_raw = image.tostring()
        label = int(labels_list[i])
        example = tf.train.Example(features=tf.train.Features(feature={'label':int64_feature(label),
                                                                       'image_raw':bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        for i in range(3):
            image = np.rot90(image)
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={'label':int64_feature(label),
                                                                       'image_raw':bytes_feature(image_raw)}))
            #show_img(image)
        
        
        
        
    writer.close()
    print('transform susccessful')

def show_img(img):
    plt.imshow(img)
    plt.show()
    
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
    
def resize_size(image_list):
    for i in range(len(image_list)):
        path = './32x32/'
        img_path = os.path.join(path,str(i)+'.jpg')
        img = cv2.imread(image_list[i],0)
        cv2.imwrite(img_path,img)

if(__name__=='__main__'):
    
    if not os.path.exists(tfrecord_train):
        print('start')
        #read_data('./image3')
        images_list,labels_list = get_data('./train')
        #resize_size(images_list)
        print(len(images_list))
        img_train,img_test,label_train,label_test = train_test_split(images_list,labels_list,test_size = 0.01, random_state=random.randint(0,100))
        convert_to_tfrecord(img_train,label_train,'./',train_file_name)
        #convert_to_tfrecord(img_test,label_test,'./',test_file_name)
        print('end')
    '''
    image,label = read_and_decode(tfrecord_train,32)
    print(image.shape,label.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        batch_x,batch_y = sess.run([image,label])
        print(batch_x.shape)
        coord.request_stop()
        coord.join(threads)
    '''
     