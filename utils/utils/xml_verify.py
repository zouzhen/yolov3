import os   #可以运行（测试水平镜像）
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw
import cv2

file_path_img = '/media/jdhl/Elements/本安安全帽数据集/JPEGImages'#图片文件夹
file_path_xml = '/media/jdhl/Elements/本安安全帽数据集/Annotations'#xml文件夹
fp=open("/media/jdhl/Elements/本安安全帽数据集/unmatch.txt","a+",encoding="utf-8")
pathDir = os.listdir(file_path_xml)
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    tree = xmlET.parse(os.path.join(file_path_xml, filename))
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)


    image_name = os.path.splitext(filename)[0]
    img = cv2.imread(os.path.join(file_path_img, image_name + '.jpg'))
    try:
        assert (height,width,depth) == img.shape
        print(image_name)
    except:
        fp.write(image_name+':'+str((height,width,depth)))
        fp.write('\n')
        size.find('width').text = str(img.shape[1])
        size.find('height').text = str(img.shape[0])
        tree.write(os.path.join(file_path_xml, filename))
        print('xml尺寸',(height,width,depth))
        print(img.shape)
fp.close()

################################################################################################
