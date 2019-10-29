import res2net
import tensorflow as tf
import numpy as np

batch_size = 1
img_size = 224

ground_truth = np.array([[[10,10,20,20,0]]])

def create_anchor(x):
    x,y = np.zeros((x.shape[0],x.shape[0],1)),np.zeros((x.shape[1],x.shape[1],1))
    for i in range(x.shape[0]):
        y[i,:] = y[i,:]+i*(img_size/x.shape[0])+(img_size/x.shape[0]/2)
        x[:,i] = x[:,i]+i*(img_size/x.shape[0])+(img_size/x.shape[0]/2)
    anchor = np.concatenate([y,x],axis=-1)
    return anchor

#def anchor_is_use(anchor_preidct,ground_truth):
#    is_use = np.zeros
#print(create_anchor(np.zeros((28,28,4))))

def gt_regression_loss(reg_predict,gt):
    anchor = create_anchor(reg_predict)/img_size
    anchor_predict = reg_predict
    ground_truth = gt/img_size
    anchor_predict[:,:,0] = anchor_predict[:,:,0]+anchor[:,:,0]
    anchor_predict[:,:,1] = anchor_predict[:,:,1]+anchor[:,:,0]
    anchor_predict[:,:,2] = anchor_predict[:,:,2]+anchor[:,:,1]
    anchor_predict[:,:,3] = anchor_predict[:,:,3]+anchor[:,:,1]
    return anchor_predict

def gt_loss_preprocess(gt,stride):
    is_fg = np.zeros((gt.shape[0],stride,stride,1))
    for i in range(gt.shape[0]):
        if((gt[2]-gt[0])*(gt[3]-gt[1])>)

 def iou_loss(gt,gt_predict):
        '''
        x = tf.placeholder(tf.float32,[None,4])
        y = tf.placeholder(tf.float32,[28,28,4])
        y_ = tf.reshape(y,[28*28,4])
        IOU_loss = tf.map_fn(lambda x:iou_loss(x,y_),elems=x)
        print(IOU_loss)
        '''
        gt_X = (gt[0]+gt[2])*(gt[1]+gt[3])
        gt_predict_X = (gt_predict[:,0]+gt_predict[:,2])*(gt_predict[:,1]+gt_predict[:,3])
        I_h = tf.minimum(gt[0],gt_predict[:,0])+tf.minimum(gt[2],gt_predict[:,2])
        I_w = tf.minimum(gt[1],gt_predict[:,1])+tf.minimum(gt[3],gt_predict[:,3])
        I = I_h*I_w
        U = gt_X*gt_predict_X
        IOU = I/U
        return IOU

with tf.Graph().as_default(): 
    '
    r = res2net.res2net(1,224,3)
    z = np.zeros((1,224,224,3))
    
    anchor_prediction = tf.placeholder(tf.float32,[batch_size,28,28,4])
    anchor_p_reshape = tf.reshape(anchor_prediction,[batch_size,28*28,4])
    anchor = tf.placeholder(tf.float32,[batch_size,None,4])
    print(r.label[...,4],anchor_prediction)
    IOU_loss = IoU(r.label[...,:4],anchor_prediction)
    print(IOU_loss)
    
    x = tf.placeholder(tf.float32,[])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        C3_reg_predict = sess.run(r.C3_reg,feed_dict={r.x:z})
        C3_anchor_preidction = np.zeros((batch_size,4))
        for b in range(batch_size):
            C3_loc = gt_regression_loss(C3_reg_predict[b],ground_truth[b])
        print(sess.run(IOU_loss,feed_dict={r.x:z,anchor_prediction:C3_loc[np.newaxis,...],r.label:ground_truth}))
        
