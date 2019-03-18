import numpy as np
import cv2


'''
6
cv2.rectangle(img,(int(dx_box_loc[i][0]), int(dx_box_loc[i][1])), 
              (int(dx_box_loc[i][2]),int(dx_box_loc[i][3])), (0, 255, 0), 2)
'''

anchor_num = 3
a_num = 1
ratio_num = 3
stride=np.array([7,14,28,56])
anchor_size=np.array([64,32,16,8])
ratio = np.array([[1,1],[1,2],[2,1]])

def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def rendom_sample(is_fg_bg,real_fg_bg_value,pos):
    remain_anchor = 256-pos
    is_bg = real_fg_bg_value[:,1]
    is_bg = np.reshape(is_bg,(is_bg.shape[0]))
    is_bg = np.where(is_bg==1)[0]
    #print(is_bg.shape)
    if(is_bg.shape[0]>remain_anchor):
        cancel_bg = np.random.choice(is_bg,is_bg.shape[0]-remain_anchor,replace=False)
        #print(cancel_bg.shape)
        real_fg_bg_value[cancel_bg,1] = 0
        is_fg_bg[cancel_bg] = 0
    
    if(pos==0):
        pos_weight = 1
    else:
        pos_weight = 128/pos
    for i in range(is_fg_bg.shape[0]):
        if(is_fg_bg[i]==1 and real_fg_bg_value[i][0]==1):
            is_fg_bg[i] = is_fg_bg[i]*pos_weight
            
    return is_fg_bg,real_fg_bg_value,pos_weight
    
class anchor_util:
    def __init__(self,feat,size):
        ratio_size = [[1,1],[1,2],[2,1]]
        self.f_num = feat.shape[0]
        self.feat = feat
        anchor0 = np.zeros([self.feat[0],self.feat[0],3,4])
        anchor1 = np.zeros([self.feat[1],self.feat[1],3,4])
        anchor2 = np.zeros([self.feat[2],self.feat[2],3,4])
        anchor3 = np.zeros([self.feat[3],self.feat[3],3,4])
        self.Anchor = {0:anchor0,1:anchor1,2:anchor2,3:anchor3}
        for i in range(self.f_num):
            a = 0
            for y in range(feat[i]):
                for x in range(feat[i]):
                    for r in range(3):
                        x1 = x*size[i]
                        y1 = y*size[i]
                        x2 = x1+(size[i]*ratio_size[r][0])
                        y2 = y1+(size[i]*ratio_size[r][1])
                        a+=1
                        self.Anchor[i][y][x][r][:] = [x1,y1,x2,y2]
            self.Anchor[i] = np.reshape(self.Anchor[i],(-1,4))
        self.Anchor = np.concatenate([self.Anchor[0],self.Anchor[1],
                                      self.Anchor[2],self.Anchor[3]],axis=0)
        #print(self.Anchor[0])
    
    def get_dx_box(self,dx):
        dx_box = np.zeros(shape=self.Anchor.shape,dtype=np.int)
        for i in range(self.Anchor.shape[0]):
            
            try:            
                anchor_cx = (self.Anchor[i][0]+self.Anchor[i][2])/2
                anchor_cy = (self.Anchor[i][1]+self.Anchor[i][3])/2
                anchor_width = self.Anchor[i][2]-self.Anchor[i][0]
                anchor_height = self.Anchor[i][3]-self.Anchor[i][1]
                
                cx = anchor_cx + anchor_width*dx[i][0]
                cy = anchor_cy + anchor_height*dx[i][1]
                cw = anchor_width*float(np.exp(dx[i][2]))
                ch = anchor_height*float(np.exp(dx[i][3]))
    
                x1 = cx-cw/2
                y1 = cy-ch/2
                x2 = x1+cw
                y2 = y1+ch
                dx_box[i][:] = x1,y1,x2,y2
            except:
                anchor_cx = (self.Anchor[i][0]+self.Anchor[i][2])/2
                anchor_cy = (self.Anchor[i][1]+self.Anchor[i][3])/2
                anchor_width = self.Anchor[i][2]-self.Anchor[i][0]
                anchor_height = self.Anchor[i][3]-self.Anchor[i][1]
                
                cx = anchor_cx + anchor_width*dx[i][0]
                cy = anchor_cy + anchor_height*dx[i][1]
                cw = anchor_width*float(np.exp(dx[i][2]))
                ch = anchor_height*float(np.exp(dx[i][3]))
                
                dx_box[i][:] = 0,0,0,0
                print(dx[i][0],dx[i][1],dx[i][2],dx[i][3])
        return dx_box
    def collect_fg_bg(self,dx_box_loc,r_box_loc):
        pos = 0
        neg = 0
        positive = False
        is_fg_bg = np.zeros([dx_box_loc.shape[0]])
        real_fg_bg_value = np.zeros([dx_box_loc.shape[0],2])
        box_index = np.zeros(shape=is_fg_bg.shape)
        max_nms = [0,0,0]
        is_anchor = 0
        #dx_box_loc = self.Anchor
        for box_num in range(r_box_loc.shape[0]):
            for i in range(dx_box_loc.shape[0]):
                result,iou = self.cal_IOU(dx_box_loc[i],r_box_loc[box_num])
                if(max_nms[0]<iou and iou>0.3):
                    max_nms[0] = iou
                    max_nms[1] = i
                    max_nms[2] = box_num
                if(result==1 and real_fg_bg_value[i][0]!=1):
                    box_index[i] = box_num
                    is_fg_bg[i] = 1
                    real_fg_bg_value[i][0] = 1
                    is_anchor+=1
                    pos+=1
                    positive=True
                elif(result==-1 and real_fg_bg_value[i][0]!=1):
                    real_fg_bg_value[i][1] = 1
                    is_fg_bg[i] = 1
                    is_anchor+=1
                    neg+=1
            
            if(positive==False and max_nms[0]>0.3):
                print(max_nms[0])
                pos+=1
                real_fg_bg_value[max_nms[1]][0] = 1
                is_fg_bg[max_nms[1]] = 1
                box_index[max_nms[1]] = max_nms[2]
                is_anchor+=1
        if(is_anchor>=256):
            is_anchor = 256
        
        is_fg_bg,real_fg_bg_value,pos_weight = rendom_sample(is_fg_bg,real_fg_bg_value,pos)
        print(pos,(is_anchor-pos))
        #show_img(img)
        is_fg_bg = is_fg_bg[np.newaxis,:]
        real_fg_bg_value = real_fg_bg_value[np.newaxis,:]
        is_anchor = is_anchor+int(pos_weight)
        return is_fg_bg,real_fg_bg_value,box_index,is_anchor
    
    def collect_anchor_dx(self,dx_box,box_index,is_fg_bg_value,r_box_loc):
        is_dx_box = np.zeros([dx_box.shape[1],4])
        is_box = 0
        dx_anchor_value = np.zeros([dx_box.shape[1],4])
        for box_num in range(r_box_loc.shape[0]):
            for i in range(is_fg_bg_value.shape[1]):
                if(is_fg_bg_value[0][i][0]==1 and box_index[i]==box_num):
                    #print('id',i)
                    r_cx = (r_box_loc[box_num][0]+r_box_loc[box_num][2])/2
                    r_cy = (r_box_loc[box_num][1]+r_box_loc[box_num][3])/2
                    r_width = (r_box_loc[box_num][2]-r_box_loc[box_num][0])
                    r_height = (r_box_loc[box_num][3]-r_box_loc[box_num][1])
                    
                    anchor_cx = (self.Anchor[i][0]+self.Anchor[i][2])/2
                    anchor_cy = (self.Anchor[i][1]+self.Anchor[i][3])/2
                    anchor_width = self.Anchor[i][2]-self.Anchor[i][0]
                    anchor_height = self.Anchor[i][3]-self.Anchor[i][1]
                    #print(r_cx,r_cy,r_width,r_height)
                    #print(anchor_cx,anchor_cy,anchor_width,anchor_height)
                    dx_anchor_value[i][0] = (r_cx-anchor_cx)/anchor_width
                    dx_anchor_value[i][1] = (r_cy-anchor_cy)/anchor_height
                    dx_anchor_value[i][2] = np.log(r_width/anchor_width)
                    dx_anchor_value[i][3] = np.log(r_height/anchor_height)
                    is_dx_box[i][:] = 1,1,1,1
                    is_box+=1
        #print(is_box)
        if(is_box==0):
            is_box=1
        is_dx_box = is_dx_box[np.newaxis,:,:]
        dx_anchor_value = dx_anchor_value[np.newaxis,:,:]
        return dx_anchor_value,is_dx_box,is_box
        
    
    def cal_IOU(self,Reframe,GTframe):
        if(Reframe[0]<0 or Reframe[1]<0 or Reframe[2]>224 or Reframe[3]>224):
            return 0,0.0
        elif(Reframe[0]>Reframe[2] or Reframe[1]>Reframe[3]):
            return 0,0.0
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
        IOU = 0
        if width <=0 or height <= 0:
            result = 0
        else:
            Area = width*height
            Area1 = width1*height1
            Area2 = width2*height2
            IOU = Area*1./(Area1+Area2-Area)
            if(IOU>=0.7):
                '''
                print(Reframe)
                cv2.rectangle(img,(int(Reframe[0]), int(Reframe[1])), (int(Reframe[2]), int(Reframe[3])), (0, 255, 0), 2)
                show_img(img)
                '''
                print(IOU)
                result = 1
            elif(IOU>=0.3):
                result = 0
            else:
                result = -1
        return result,IOU
    
    def get_best_anchor(self,pfi,pfs,dx_box_loc,num=10):
        loc = []
        for i in range(num):
            p_index = pfi[i]
            x1,y1,x2,y2 = dx_box_loc[p_index]
            #print(p_index)
            #print(p_index,x1,y1,x2,y2)
            if(x1<0 or y1<0 or x2>224 or y2>224 or x1>x2 or y1>y2):
                continue
            #print(p_index)
            loc.append([x1,y1,x2,y2,pfs[i],p_index])
        return np.array(loc)
    
    def py_cpu_nms(self,dets,thresh=0.15):  
        """Pure Python NMS baseline."""  
        x1 = dets[:, 0]  
        y1 = dets[:, 1]  
        x2 = dets[:, 2]  
        y2 = dets[:, 3]  
        scores = dets[:, 4]  
        order = scores.argsort()[::-1]  
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
        print(order)
        keep = []  
        while order.size > 0:  
     
            i = order[0]  
            keep.append(i)  
    
            xx1 = np.maximum(x1[i], x1[order[1:]])  
            yy1 = np.maximum(y1[i], y1[order[1:]])  
            xx2 = np.minimum(x2[i], x2[order[1:]])  
            yy2 = np.minimum(y2[i], y2[order[1:]])  
      
            w = np.maximum(0.0, xx2 - xx1 + 1)  
            h = np.maximum(0.0, yy2 - yy1 + 1)  
            inter = w * h  
    
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  
      
            inds = np.where(ovr <= thresh)[0]  
    
            order = order[inds + 1]  
        keep = np.array(keep)
        return keep
    
'''
feat = np.array([7,14,28,56])
size = np.array([32,16,8,4])
a = anchor_util(feat,size)
'''