import numpy as np
import cv2


'''
cv2.rectangle(img,(int(dx_box_loc[i][0]), int(dx_box_loc[i][1])), 
              (int(dx_box_loc[i][2]),int(dx_box_loc[i][3])), (0, 255, 0), 2)
'''
def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

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
        
        return dx_box
    def collect_fg_bg(self,dx_box_loc,r_box_loc):
        img = cv2.imread('test.jpg')
        pos = 0
        positive = False
        is_fg_bg = np.zeros([dx_box_loc.shape[0]])
        real_fg_bg_value = np.zeros([dx_box_loc.shape[0],2])
        box_index = np.zeros(shape=is_fg_bg.shape)
        max_nms = [0,0]
        is_anchor = 0
        for box_num in range(r_box_loc.shape[0]):
            for i in range(dx_box_loc.shape[0]):
                result,iou = self.cal_IOU(dx_box_loc[i],r_box_loc[box_num])
                if(max_nms[0]<iou):
                    max_nms[0] = iou
                if(result==1 and real_fg_bg_value[i][0]!=1):
                    box_index[i][0] = i
                    is_fg_bg[i] = 1
                    real_fg_bg_value[i][0] = 1
                    is_anchor+=1
                    pos+=1
                    positive=True
                elif(result==-1 and real_fg_bg_value[i][0]!=1):
                    real_fg_bg_value[i][1] = 1
                    is_fg_bg[i] = 1
                    is_anchor+=1
            
            if(positive==False):
                pos+=1
                real_fg_bg_value[max_nms[1]][0] = 1
                is_fg_bg[max_nms[1]] = 1
                is_anchor+=1
        
        t = np.where(real_fg_bg_value>0)
        print(t)
        print(t[0])
        print(is_anchor,pos)
        #show_img(img)
        return is_fg_bg,real_fg_bg_value,box_index,is_anchor
    
    def collect_anchor_dx(self,dx_box,box_index,is_fg_bg_value,r_box_loc):
        is_dx_box = np.zeros(shape=dx_box.shape)
        is_box = 0
        dx_anchor_value = np.zeros(shape=dx_box.shape)
        for box_num in range(r_box_loc.shape[0]):
            for i in range(len(is_fg_bg_value)):
                if(is_fg_bg_value[i][0]==1 and box_index[i]==box_num):
                    r_cx = (r_box_loc[box_num][0]+r_box_loc[box_num][2])/2
                    r_cy = (r_box_loc[box_num][1]+r_box_loc[box_num][3])/2
                    r_width = (r_box_loc[box_num][2]-r_box_loc[box_num][0])
                    r_height = (r_box_loc[box_num][3]-r_box_loc[box_num][1])
                    
                    anchor_cx = (self.Anchor[i][0]+self.Anchor[i][2])/2
                    anchor_cy = (self.Anchor[i][1]+self.Anchor[i][3])/2
                    anchor_width = self.Anchor[i][2]-self.Anchor[i][0]
                    anchor_height = self.Anchor[i][3]-self.Anchor[i][1]
                    
                    dx_anchor_value[i][0] = (r_cx-anchor_cx)/anchor_width
                    dx_anchor_value[i][1] = (r_cy-anchor_cy)/anchor_height
                    dx_anchor_value[i][2] = np.log(r_width/anchor_width)
                    dx_anchor_value[i][3] = np.log(r_height/anchor_height)
                    is_dx_box[i][:] = 1,1,1,1
                    is_box+=1
        print(is_box)
        return is_dx_box,is_box
        
    
    def cal_IOU(self,Reframe,GTframe):
        if(Reframe[0]<0 or Reframe[1]<0 or Reframe[2]>224 or Reframe[3]>224):
            return 0,0.0
        elif(Reframe[0]>=Reframe[2] or Reframe[1]>=Reframe[3]):
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
                print(IOU)
                result = 1
            elif(IOU>=0.3):
                result = 0
            else:
                result = -1
        return result,IOU
    
'''
feat = np.array([7,14,28,56])
size = np.array([32,16,8,4])
a = anchor_util(feat,size)
'''