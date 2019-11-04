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
            if type(self.food) == list and len(self.food) != 0:
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

    def stream(self, imgs_path, item, save_path):
    	image = cv2.imread(os.path.join(imgs_path, item), 1)
    	frame = cv2.resize(image, (1920,1080), interpolation=cv2.INTER_CUBIC)
    	cv2.imwrite(save_path + '/' +item.split('.')[0] + '.jpg', frame)

    def run(self):  # call start()时 就会调用run(run为单进程).
        while True:
            item = self.queue.get()
            print("Consumer-->%s" % item, self.label)
            self.stream(self.args['imgs_path'],item,self.args['save_path'])
            self.queue.task_done()


if __name__ == '__main__':
	imgs_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/需要修改的图片'
	save_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/修改后的图片'
	pathlist = os.listdir(imgs_path)
	for file in pathlist:
		if os.path.isdir(file):
			pathlist.remove(file)

	os.makedirs(save_path)
	# 统计计算内部的核心进程数
	cores = multiprocessing.cpu_count()
	qMar = Manager()
	# 取核心进程数的一半建立数据队列
	q1 = qMar.Queue(10)
	# q1 = qMar.Queue(cores-2)
	p = Producer(q1, pathlist)
	processes = []
	processes.append(p)
	# print(int(cores/2))
	for i in range(10):
	# for i in range(cores-2):
		processes.append(Consumer(q1,i,imgs_path=imgs_path,save_path=save_path))
		
	[process.start() for process in processes]
	[process.join() for process in processes]

