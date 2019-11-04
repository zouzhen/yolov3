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
        tree = xmlET.parse(os.path.join(xml_path, item))
        objs = tree.findall('object')
        num_objs = len(objs)

        if num_objs >= 0:
            boxes = np.zeros((num_objs, 5), dtype=np.uint16)
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)

                label = obj.find('name').text
                if label == 'Nonstandard':
                    label = float(1)
                if label == 'Standard':
                    label = float(0)
                else:
                    label = float(2)    


                boxes[ix, 0:5] = [x1, y1, x2, y2, label]


            image_name = os.path.splitext(item)[0]
            img = Image.open(os.path.join(imgs_path, image_name + '.jpg'))

            draw = ImageDraw.Draw(img)
            for ix in range(len(boxes)):
                xmin = int(boxes[ix, 0])
                ymin = int(boxes[ix, 1])
                xmax = int(boxes[ix, 2])
                ymax = int(boxes[ix, 3])
                label = int(boxes[ix, 4])
                # if label == 1:
                    # draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
                # if label == 0:
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0))

            img.save(os.path.join(save_path, image_name + '.jpg'))

    def run(self):  # call start()时 就会调用run(run为单进程).
      while True:
          item = self.queue.get()
          self.horizontal_mirror_imgs(self.args['imgs_path'],self.args['xml_path'],item,self.args['save_path'])
          print("Consumer-->%s" % item, self.label)
          self.queue.task_done()


if __name__ == '__main__':
	imgs_path = '/media/jdhl/Elements/yibiaoshibie/darknet/scripts/VOCdevkit/VOC2007/JPEGImages'
	xml_path = '/media/jdhl/Elements/yibiaoshibie/darknet/scripts/VOCdevkit/VOC2007/Annotations'
	save_path = '/media/jdhl/Elements/yibiaoshibie/darknet/scripts/VOCdevkit/VOC2007/result'
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