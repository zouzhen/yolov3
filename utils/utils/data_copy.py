import os
import shutil
'''
本脚本实现了对不同文件的脚本
'''

number = 10000
ori_file = '/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/yolov3/data/VOCdevkit/VOC2012/webwxgetmsgimg.jpg'
ori_xml = '/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/yolov3/data/VOCdevkit/VOC2012/C3X17RBM502B21345DD+B+CFC+CP03XRBMXTS10002.xml'
tar_ImageFile = '/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/yolov3/data/VOCdevkit/VOC2012/JPEGImages/'
tar_XmlFile = '/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/yolov3/data/VOCdevkit/VOC2012/Annotations/'


for name in range(number):
    print('%08d' % name)
    # shutil.copy2(ori_file,tar_ImageFile + '%08d' % name + '.jpg')
    shutil.copy2(ori_xml,tar_XmlFile + '%08d' % name + '.xml')
