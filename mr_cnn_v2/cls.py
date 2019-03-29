import os
import cv2
import numpy as np

def write_data(path,save_path):
    a = 0
    for root,dirs,files in os.walk(path):
        for f in files:
            if(f.endswith('bmp')):
                try:
                    img_path = os.path.join(root,f)
                    img = cv2.imread(img_path)
                    '''
                    cv2.imshow('img',img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    '''
                    img = cv2.resize(img,(1172,863))
                    jpg_name = str(a)+'.bmp'
                    img_path = os.path.join(save_path,jpg_name)
                    cv2.imwrite(img_path,img)
                    a+=1
                except:
                    print(a)


if(__name__=='__main__'):
    a = 0
    write_data('./bu6','./train')
    #write_data('./Good_unit/QUALCOMM','./good')
    
                
                