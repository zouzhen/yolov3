'''
此脚本解决啦
'''

#!/usr/bin/env python
# coding=utf-8
import os
#利用文件名字长排序删除写错的xml文件
#path = 'C:\\Users\\liuzhichao\\Desktop\\着装25-29'
#path = 'C:\\Users\\liuzhichao\\Desktop\\VOC2007\\202_h'
path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/7-02/仪表标注数据/仪表训练集/JPEGImages'

filelist = os.listdir(path)
i = 0
for filename in filelist:
    if '_zym' in filename:
        i=i+1
        print(i)
        print(filename)

        #os.remove(r'C:\\Users\\liuzhichao\\Desktop\\VOC2007\\202_h\\' + filename)  #####删除没有目标图片文件
        os.remove(r'/home/jdhl/WorkSpace/ZOUZHEN/dataset/7-02/仪表标注数据/仪表训练集/JPEGImages/' + filename)  #####删除没有目标图片文件