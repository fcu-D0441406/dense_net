import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import cv2
import resnet_v2

save_ckpt = './checkpoint1_dir\\checkpoint'
save_meta = './checkpoint1_dir\\MyModel'
CLASS_NUM = 1
box_num = 3
rs_num = 3
box = [256,128,64]
rate_size = [[1,1],[1,2],[2,1]]
threshold = 0.7
initial_lr = 2e-4
def share_net(image):
    resnet = resnet_v2.ResNet(image)
    s_net = resnet.net3
    s_net = batch_norm(s_net)
    return s_net

def batch_norm(inputs,is_training=True,is_conv_out=True,decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
    pop_var = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
    if(is_training):
        if(is_conv_out):
            batch_mean,batch_var=tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean,batch_var=tf.nn.moments(inputs,[0])
            
        train_mean = tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
        train_var = tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.001)

def rpn_net(s_net):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        r_net = slim.conv2d(s_net,512,[3,3],scope='rpn_conv')
        rpn_cls = slim.conv2d(r_net,2*box_num*rs_num,[1,1],scope='cls_score')
        rpn_cls = tf.reshape(rpn_cls,[-1,14,14,1*box_num*rs_num,2])
        rpn_cls = tf.nn.softmax(rpn_cls)
        rpn_box = slim.conv2d(r_net,4*box_num*rs_num,[3,3],scope='rpn_box')
        rpn_box = tf.reshape(rpn_box,[-1,14,14,box_num*rs_num,4])
    return rpn_cls,rpn_box


def IOU(Reframe,GTframe):
    if(GTframe[0]<0 or GTframe[1]<0 or GTframe[2]>800 or GTframe[3]>600):
        return 0
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)
    if width <=0 or height <= 0:
        ratio = -1
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
        #print(ratio)
        if(ratio>=threshold):
            ratio = 1
        elif(ratio>=0.3):
            ratio = 0
        else:
            ratio = -1
    return ratio

def cal_iou(predict_box,result):
    result_iou = np.zeros(shape=[14,14,1*box_num*rs_num,2],dtype=np.float64)
    result_box = [result[2],result[3],result[4],result[5]]
    predict_box = predict_box[0]
    is_iou = 0
    for i in range(14):
        for j in range(14):
            for z in range(box_num):
                for z2 in range(rs_num):
                    p_box = [j*57-(box[z]*rate_size[z2][0]/2),i*42-(box[z]*rate_size[z2][1]/2),j*57+(box[z]*rate_size[z2][0]/2),i*42+(box[z]*rate_size[z2][1]/2)]
                    #result_iou[i][j]=IOU(result_box,p_box)
                    if(IOU(result_box,p_box)==1):
                        result_iou[i][j][z*rs_num+z2][0]=1
                        is_iou = is_iou+1                        
                        #print(i,j,z*rs_num+z2)
                        #print(p_box)
                        #print('-----')                        
                    elif(IOU(result_box,p_box)==-1):
                        result_iou[i][j][z*rs_num+z2][1]=1
                        is_iou = is_iou+1
                    #print(result_box,p_box)
                    #print(IOU(result_box,p_box))
    return result_iou,is_iou


    

def rpn_train(x,rpn_cls,rpn_box,result,input_img=None):
    #先預測[14,14]個方塊,預測哪個方塊為前景,再用iou(實際,預測),如果大於threshold就預測為前景
    #前景預測完後,再用box_regression微調預測完前景方塊的(x,y,w,h)最後預測出物體的(x,y,w,h)
    Is_iou = tf.placeholder(tf.float32)
    Is_box = tf.placeholder(tf.float32)
    IOU = tf.placeholder(tf.float32,[None,14,14,box_num*rs_num,2])
    bounding_box = tf.placeholder(tf.float32,[None,14,14,box_num*rs_num,4])
    #loss = (-tf.reduce_sum(IOU*tf.log(tf.clip_by_value(rpn_cls,1e-8,1.0)),reduction_indices=[1]))# loss
    #loss = tf.div(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=IOU,logits=rpn_cls)),Is_iou)
    loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits_v2(labels=IOU,logits=rpn_cls)))
    
    sm1_sub = tf.add(bounding_box,-1*rpn_box)#loss2 使用smooth L1作法
    sm1_sub = tf.abs(sm1_sub)
    sm1_bool = tf.cast(tf.less_equal(sm1_sub,1.0),tf.float32)
    sm1_bool2 = tf.cast(tf.equal(bounding_box,0),tf.float32)
    print(rpn_box,bounding_box,sm1_sub,sm1_bool,sm1_bool2)
    #loss2 = (tf.reduce_sum((1-sm1_bool2)*(sm1_bool*(0.5*sm1_sub*sm1_sub)+(1-sm1_bool)*(sm1_sub-0.5))))
    loss2 = tf.reduce_mean((1-sm1_bool2)*(sm1_bool*(0.5*sm1_sub*sm1_sub)+(1-sm1_bool)*(sm1_sub-0.5)))
    
    total_loss = tf.add(loss,loss2)
    optimizer = tf.train.AdamOptimizer(initial_lr).minimize(total_loss)
    #ls_optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        if not os.path.exists(save_ckpt):
            for i in range(300):
                predict_box,archor_box = sess.run([rpn_cls,rpn_box],feed_dict={x:input_img})
                result_iou,is_iou = cal_iou(predict_box,result)
                up_box,is_box = box_regression(archor_box,result_iou,result)
                up_box = up_box[np.newaxis,:,:,:,:]
                result_iou = result_iou[np.newaxis,:,:,:,:]
                _,total_ls,ls,ls2 = sess.run([optimizer,total_loss,loss,loss2],feed_dict={Is_iou:is_iou,x:input_img,
                                             IOU:result_iou,bounding_box:up_box,Is_box:is_box})
                #_,ls = sess.run([ls_optimizer,loss],feed_dict={x:input_img,IOU:result_iou,bounding})
                #_,ls = sess.run([ls_optimizer,loss],feed_dict={x:input_img,IOU:result_iou,Is_iou:is_iou})
                if(i%10==0):
                    #print(ls,predict_box[0][7][7][0],result_iou[0][7][7][0])
                    print('step',i,' ',total_ls,' ',ls,' ',ls2,predict_box[0][7][7][0],result_iou[0][7][7][0])
            iou,ar_box = sess.run([rpn_cls,rpn_box],feed_dict={x:input_img})
            print(iou.shape,ar_box.shape)
            draw_image(iou[0],ar_box[0])
            #saver.save(sess,save_meta)
        else:
            saver = tf.train.import_meta_graph('./checkpoint1_dir/MyModel.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./checkpoint1_dir'))
            graph = tf.get_default_graph()
            rpn_cls = graph.get_tensor_by_name('Reshape_3:0')
            rpn_box = graph.get_tensor_by_name('Reshape_4:0')
            iou,ar_box = sess.run([rpn_cls,rpn_box],feed_dict={x:input_img})
            draw_image(iou[0],ar_box[0])

def box_regression(archor_box,result_iou,result):
    up_box = np.ones([14,14,box_num*rs_num,4],dtype=np.float32)
    #up_box = np.zeros([14,14,box_num*rs_num,4],dtype=np.float32)
    is_box = 0
    for i in range(14):
        for j in range(14):
            for z in range(box_num):
                for z2 in range(rs_num):
                    if(result_iou[i][j][z*rs_num+z2][0]>=threshold):
                        r_loc = [int((result[2]+result[4])/2),int((result[3]+result[5])/2),int(result[4]-result[2]),int(result[5]-result[3])]#預測使用中心位置(x,y)/寬/高
                        p_loc = [int(j*57),int(i*42),box[z]*rate_size[z2][0],box[z]*rate_size[z2][1]]#預測使用中心位置(x,y)/寬/高
                        up_box[i][j][z*rs_num+z2][0] = (r_loc[0]-p_loc[0])/p_loc[2]
                        up_box[i][j][z*rs_num+z2][1] = (r_loc[1]-p_loc[1])/p_loc[3]
                        up_box[i][j][0][2] = np.log(r_loc[2]/p_loc[2])
                        up_box[i][j][0][3] = np.log(r_loc[3]/p_loc[3])
                        #up_box[i][j][z*rs_num+z2][2] = (r_loc[2]/p_loc[2])
                        #up_box[i][j][z*rs_num+z2][3] = (r_loc[3]/p_loc[3])
                        is_box = is_box+1
                        '''
                        print(up_box[i][j][0])
                        print('-----')
                        '''
    return up_box,is_box

def show_archor():
    img = cv2.imread('./img.jpg')
    for i in range(14):
        for j in range(14):
            for z in range(box_num):
                for z2 in range(rs_num):
                    if(i==7 and j==8):
                        x1 = int(j*57-(box[z]*rate_size[z2][0]/2))
                        y1 = int(i*42-(box[z]*rate_size[z2][1]/2))
                        x2 = int(j*57+(box[z]*rate_size[z2][0]/2))
                        y2 = int(i*42+(box[z]*rate_size[z2][1]/2))
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0, 255, 0), 2)
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
                    
def draw_image(iou,ar_box):
    img = cv2.imread('./img.jpg')
    for i in range(14):
        for j in range(14):
            for z in range(box_num):
                for z2 in range(rs_num):
                    if(iou[i][j][z*rs_num+z2][0]>=threshold):
                        #cv2.rectangle(img, (int(j*57-box/2),int(i*42-box/2)),(int(j*57+box/2),int(i*42+box/2)), (0, 255, 0), 2)
                        c_x = int(j*57+(box[z]*rate_size[z2][0]*ar_box[i][j][z][0]))
                        c_y = int(i*42+(box[z]*rate_size[z2][1]*ar_box[i][j][z][1]))
                        w = int((box[z]*rate_size[z2][0]*np.exp(ar_box[i][j][z][2])))
                        h = int((box[z]*rate_size[z2][0]*np.exp(ar_box[i][j][z][3])))
                        #w = int((box[z]*rate_size[z2][0]*ar_box[i][j][z][2]))
                        #h = int((box[z]*rate_size[z2][1]*ar_box[i][j][z][3]))
                        cv2.rectangle(img, (int(c_x-(w/2)),int(c_y-(h/2))),(int(c_x+(w/2)),int(c_y+(h/2))), (0, 255, 0), 2)
                        print(j,i,ar_box[i][j][z][0],ar_box[i][j][z][1],ar_box[i][j][z][2],ar_box[i][j][z][3])
                        cv2.imshow('img',img)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

def read_image(path):
    input_img = cv2.imread(path)
    input_img = cv2.resize(input_img,(224,224))
    input_img = np.reshape(input_img,(1,224,224,3))
    return input_img

if(__name__=='__main__'):
    
    x = tf.placeholder(tf.float32, [None, 224,224,3])
    image = tf.reshape(x, [-1, 224, 224, 3])
    result = ['./img.jpg','cabe',264,158,611,422]
    input_img = read_image(result[0])
    s_net = share_net(image)
    #print(s_net)
    rpn_cls,rpn_box = rpn_net(s_net)
    rpn_train(x,rpn_cls,rpn_box,result,input_img)
    
    #show_archor()

