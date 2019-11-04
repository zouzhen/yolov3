##############python批量修改xml的bounding box数值，修改为图片镜像翻转之后的包围框坐标###########################
# coding:utf-8
import cv2
#import math
import numpy as np
import xml.etree.ElementTree as ET
import os

#xmlpath = 'G:\\face_xym_xml\\'
#xmlpath = 'E:\\face_zym\\Annotations_zym\\'
#xmlpath = 'D:\\安全帽数据0531原始\\helmet_xml=0\\'#xml文件原始路径#################根据xml文件制作镜像xml文件
#xmlpath = 'C:\\Users\\liuzhichao\\Desktop\\着装xml总\\'
xmlpath ='/home/jdhl/WorkSpace/ZOUZHEN/dataset/7-02/仪表标注数据/c_xml/'
#rotated_imgpath = './rotatedimg/'#镜像图片路径
#rotated_xmlpath = 'D:\\安全帽数据0531原始\\helmet_xml_zym1\\'#镜像xml路径
#rotated_xmlpath = 'D:\\安全帽数据0531原始\\helmet_xml_zym1\\'#镜像xml路径
rotated_xmlpath = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/7-02/仪表标注数据/xml镜像/'#镜像xml路径
#rotated_xmlpath = 'G:\\face_xym_xml\\'

for i in os.listdir(xmlpath):
    a, b = os.path.splitext(i)
    print(str(i))
    tree = ET.parse(xmlpath + a + '.xml')
    root = tree.getroot()
    #oldname= root.getElementsByTagName('filename')
    for name in root.iter('filename'):#改一下名字
        #older_name = str(name.text)
        new_name = a + '_zym.jpg'
        name.text = str(new_name)
    #print("oldname", oldname)
    for chi in root.iter('size'):
        width = int(chi.find('width').text)
        #print("width",width)
    for box in root.iter('bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)


        box.find('xmin').text = str(width - xmax)
        box.find('ymin').text = str(ymin)

        box.find('xmax').text = str(width - xmin)
        box.find('ymax').text = str(ymax)
    tree.write(rotated_xmlpath + a + '_zym.xml')
        #print(str(a) + '.xml has been rotated for  ' + '°')
