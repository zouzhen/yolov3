'''
该脚本实现了生产者消费者模型
以多进程或者多线程的方式
'''

import cv2
import os
import sys
import copy
import time
import multiprocessing
from multiprocessing import Process, Manager
import os.path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw
from datetime import datetime


class Producer(Process):

    def __init__(self, queue, food):  # 重写.
        super().__init__()  # 加入父类init.
        self.queue = queue
        self.food = food

    def run(self):  # call start()时 就会调用run(run为单进程).
        while True:
        # print('1')
            if type(self.food) == list:
                if len(self.food)==0:
                    print("生产者生产完毕")
                    sys.exit()                
                item = self.food.pop()  # left is closed and right is closed.
                self.queue.put(item)
                print("Producer-->%s" % item)
                time.sleep(0.1)



class Consumer(Process):
    def __init__(self, queue, label, **args):  # 重写.
        super().__init__()  # 加入父类init.
        self.queue = queue
        self.label = label
        self.args = args

    def horizontal_mirror_imgs(self, imgs_path, xml_path, item, save_path):
        tree = xmlET.parse(os.path.join(xml_path, item))
        root = tree.getroot()

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            x2 = float(bbox.find('xmax').text)
            y1 = float(bbox.find('ymin').text)
            y2 = float(bbox.find('ymax').text)

            if abs(x2 - x1) < 20 or abs(y1 - y2) < 20:
                root.remove(obj)
        tree.write(os.path.join(save_path, item))

 

    def run(self):  # call start()时 就会调用run(run为单进程).
      while True:
          item = self.queue.get()
          self.horizontal_mirror_imgs(self.args['imgs_path'],self.args['xml_path'],item,self.args['save_path'])
          print("Consumer-->%s" % item, self.label)
          self.queue.task_done()


if __name__ == '__main__':
	imgs_path = '/media/jdhl/Elements/本安安全帽数据集/JPEGImages'
	xml_path = '/media/lzc274500/Elements SE/莱西风电数据集/11-14/dataset/Annotations'
	save_path = '/media/lzc274500/Elements SE/莱西风电数据集/11-14/dataset/Annotations_correction'
	pathlist = os.listdir(xml_path)
	# 统计计算内部的核心进程数
	if not os.path.exists(save_path):   
	  os.makedirs(save_path)
	cores = multiprocessing.cpu_count()
	qMar = Manager()
	# 取核心进程数的一半建立数据队列
	q1 = qMar.Queue(cores-5)
	p = Producer(q1, pathlist)
	processes = []
	processes.append(p)
	# print(int(cores/2))
	for i in range(cores-5):
		processes.append(Consumer(q1,i,imgs_path=imgs_path,xml_path=xml_path, save_path=save_path))
		
	[process.start() for process in processes]
	[process.join() for process in processes]