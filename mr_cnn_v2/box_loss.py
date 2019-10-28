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

def IoU(y_pred, y_true):
    I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tf.reduce_mean(I / U)

def get_iou(masks, predictions):
    ious = []
    for i in range(batch_size):
        mask = masks[i]
        pred = predictions[i]
        masks_sum = tf.reduce_sum(mask)
        predictions_sum = tf.reduce_mean(pred)
        intersection = tf.reduce_sum(tf.multiply(mask, pred))
        union = masks_sum + predictions_sum - intersection
        iou = intersection / union
        ious.append(iou)
    return ious

def IOU(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = tf.maximum(x1+width1,x2+width2)
    startx = tf.minimum(x1,x2)
    width = width1+width2-(endx-startx)

    endy = tf.maximum(y1+height1,y2+height2)
    starty = tf.minimum(y1,y2)
    height = height1+height2-(endy-starty)
    
    


    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
    return ratio



with tf.Graph().as_default(): 
    r = res2net.res2net(1,224,3)
    z = np.zeros((1,224,224,3))
    
    anchor_prediction = tf.placeholder(tf.float32,[batch_size,28,28,4])
    anchor_p_reshape = tf.reshape(anchor_prediction,[batch_size,28*28,4])
    anchor = tf.placeholder(tf.float32,[batch_size,None,4])
    print(r.label[...,4],anchor_prediction)
    IOU_loss = IoU(r.label[...,:4],anchor_prediction)
    print(IOU_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        C3_reg_predict = sess.run(r.C3_reg,feed_dict={r.x:z})
        C3_anchor_preidction = np.zeros((batch_size,4))
        for b in range(batch_size):
            C3_loc = gt_regression_loss(C3_reg_predict[b],ground_truth[b])
        print(sess.run(IOU_loss,feed_dict={r.x:z,anchor_prediction:C3_loc[np.newaxis,...],r.label:ground_truth}))
