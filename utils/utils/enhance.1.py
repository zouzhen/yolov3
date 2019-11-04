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


class Producer(Process):

    def __init__(self, queue, food):  # 重写.
      super().__init__()  # 加入父类init.
      self.queue = queue
      self.food = food

    def run(self):  # call start()时 就会调用run(run为单进程).
      while True:
        # print('1')
        if type(self.food) == list:
          item = self.food.pop()  # left is closed and right is closed.
          self.queue.put(item)
          print("Producer-->%s" % item)
          time.sleep(0.1)
          if len(self.food)==0:
            print("生产者生产完毕")


class Consumer(Process):
    def __init__(self, queue, label, **args):  # 重写.
      super().__init__()  # 加入父类init.
      self.queue = queue
      self.label = label
      self.args = args

    def horizontal_mirror_imgs(self, imgs_path, item, save_path):
      image = cv2.imread(os.path.join(imgs_path, item), 1)
      height = image.shape[0]
      width = image.shape[1]
      # channels = image.shape[2]
      iLR = copy.deepcopy(image)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制

      for i in range(height):
        for j in range(width):
          iLR[i, width - 1 - j] = image[i, j]
      # cv2.imshow('image', image)
      # cv2.imshow('iLR', iLR)
      save_name = item[:-4]+'_zym'+'.jpg'

      cv2.imwrite(os.path.join(save_path, save_name), iLR,
          [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 保存图片

    def run(self):  # call start()时 就会调用run(run为单进程).
      while True:
          item = self.queue.get()
          self.horizontal_mirror_imgs(self.args['imgs_path'],item,self.args['save_path'])
          print("Consumer-->%s" % item, self.label)
          self.queue.task_done()


if __name__ == '__main__':
	imgs_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/数据集镜像/stream'
	save_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/数据集镜像/stream2'
	pathlist = os.listdir(imgs_path)
	# 统计计算内部的核心进程数
	if not os.path.exists(save_path):
	  os.makedirs(save_path)
	cores = multiprocessing.cpu_count()
	qMar = Manager()
	# 取核心进程数的一半建立数据队列
	q1 = qMar.Queue(cores-2)
	p = Producer(q1, pathlist)
	processes = []
	processes.append(p)
	# print(int(cores/2))
	for i in range(cores-2):
		processes.append(Consumer(q1,i,imgs_path=imgs_path,save_path=save_path))
		
	[process.start() for process in processes]
	[process.join() for process in processes]