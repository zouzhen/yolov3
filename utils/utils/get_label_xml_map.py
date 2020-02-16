'''
该脚本实现了生产者消费者模型
以多进程或者多线程的方式
'''

import cv2
import os
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
import random


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
                    break                
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
        imges = cv2.imread(os.path.join(imgs_path,item.split(".")[0]+'.jpg'))
        tree = xmlET.parse(os.path.join(xml_path, item))
        root = tree.getroot()
        flag = 0
        for index,obj in enumerate(root.findall('object')):
            # obj.find('name').text = 'Person'
            if obj.find('name').text in ['Back']:
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = int(bbox.find('xmin').text)
                x2 = int(bbox.find('xmax').text)
                y1 = int(bbox.find('ymin').text)
                y2 = int(bbox.find('ymax').text)
                # position = self.random_change([x1,x2,y1,y2])
                img = imges[y1:y2, x1:x2]
                # img = cv2.resize(img,img_size[obj.find('name').text],interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(save_path,item.split(".")[0] + str(index) + '.jpg'),img)
            else:
                pass
        # tree.write(os.path.join(save_path, item))
    def random_change(self,position:list):
        ratio = random.uniform(-0.25,0.25)
        distance = (position[1]-position[0]) * ratio/2
        position[0] = int(position[0] - distance) if (position[0] - distance) > 0 else 0
        position[1] = int(position[1] + distance) if (position[1] + distance) > 0 else 0
        distance = (position[3]-position[2]) * ratio/2
        position[2] = int(position[2] - distance) if (position[2] - distance) > 0 else 0
        position[3] = int(position[3] + distance) if (position[3] + distance) > 0 else 0
        
        return position

 

    def run(self):  # call start()时 就会调用run(run为单进程).
      while True:
          item = self.queue.get()
          self.horizontal_mirror_imgs(self.args['imgs_path'],self.args['xml_path'],item,self.args['save_path'])
          print("Consumer-->%s" % item, self.label)
          self.queue.task_done()


if __name__ == '__main__':
	imgs_path = 'F:\\LabelImg\\Data\\JPEGImages'
	xml_path = 'F:\\LabelImg\\Data\\Annotations'
	save_path = 'F:\\LabelImg\\Data\\getresult'
	pathlist = os.listdir(xml_path)
	# 统计计算内部的核心进程数
	if not os.path.exists(save_path):
	  os.makedirs(save_path)
	cores = multiprocessing.cpu_count()
	qMar = Manager()
	# 取核心进程数的一半建立数据队列
	q1 = qMar.Queue(cores-3)
	p = Producer(q1, pathlist)
	processes = []
	processes.append(p)
	# print(int(cores/2))
	for i in range(cores-3):
		processes.append(Consumer(q1,i,imgs_path=imgs_path,xml_path=xml_path, save_path=save_path))
		
	[process.start() for process in processes]
	[process.join() for process in processes]