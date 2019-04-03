import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import cv2

cls_idx = {'fg':0}

def read_xml(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        #print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_data = [root.find('path').text]
        for member in root.findall('object'):
            value = (
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_data.append(value)
        xml_list.append(xml_data)
    #print(xml_list)
    return xml_list

def decode_xml(xml_list):
    loc = np.zeros([len(xml_list)-1,4],dtype=np.float32)
    img = cv2.imread(xml_list[0])
    cls = []
    for i in range(1,len(xml_list)):
        cls.append(cls_idx[xml_list[i][0]])
        loc[i-1] = xml_list[i][1:]
    loc = np.reshape(np.array(loc),(-1,4))
    return img,loc,cls
'''
def train():
    image_path = os.path.join(os.getcwd(), 'data', 'tf_wider_train', 'annotations','xmls')
    xml_df = xml_to_csv(image_path)
    labels_path = os.path.join(os.getcwd(), 'data', 'tf_wider_train','train.csv')
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_train - Successfully converted xml to csv.')
 
def val():
    image_path = os.path.join(os.getcwd(), 'data', 'tf_wider_val', 'annotations','xmls')
    xml_df = xml_to_csv(image_path)
    labels_path = os.path.join(os.getcwd(), 'data', 'tf_wider_val', 'val.csv')
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_val -  Successfully converted xml to csv.')
'''
'''
#train()
#val()
xml_list = xml_to_csv('./xml_test')
img,loc = read_xml(xml_list[0])
print(loc.shape)
#print((xml_list))
'''
