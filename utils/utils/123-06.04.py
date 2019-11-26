'''
该脚本实现了生产者消费者模型
以多进程或者多线程的方式
'''

import cv2
import os
import sys
import copy
import time
from datetime import datetime
import numpy as np
import random
import multiprocessing as mp
from ctypes import *
import requests
from multiprocessing import Process, Manager, JoinableQueue

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("row", c_int),
                ("col", c_int),
                ("mask_num", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("/home/lzc274500/face_detect_multiple/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

float_to_image = lib.float_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

float_to_image = lib.float_transfer_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

net = load_net("cfg/yolov3-tiny.cfg".encode("utf-8"), "face_detect_140000.weights".encode("utf-8"), 0,1)
meta = load_meta("cfg/voc.data".encode("utf-8"))

def upload_event_message(url, camera_ip, event_type, timestamp, name, photo_information):
    requests.post(url, data={'camera_ip':camera_ip,'event_type':event_type,'timestamp':timestamp,'name':name,'photo_information':photo_information})
    return True

def detect(im, thresh=0.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    im = im.astype('float32')
    #net = load_net("cfg/yolov3-tiny.cfg".encode("utf-8"), "face_detect_140000.weights".encode("utf-8"), 0)
    #meta = load_meta("cfg/voc.data".encode("utf-8"))
    data = cast(im.ctypes.data, POINTER(c_float))
    im = float_to_image(im.shape[1], im.shape[0], 3, data)
    predict_image(net, im)

    dets = get_network_boxes(net, im.w, im.h,thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]

    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        cla_prob = []
        for ii in range(meta.classes):
            cla_prob.append(dets[j].prob[ii])

        if sum(cla_prob) != 0:
            i = np.argmax(cla_prob)
            print("i",i)
            b = dets[j].bbox
            print("b",b)

            x = b.x
            y = b.y
            w = b.w
            h = b.h

            

            left = int(x - w / 2)
            right = int(x + w / 2)
            top = int(y) - int((h)/2)
            bot = int(y + h/2)

            #if py_version == 3:
                                                                # (  x1,  y1,    x2,  y2)
            #res.append((meta.names[i].decode('utf-8'), dets[j].prob[i], (left, top, right, bot)))
            
            res.append((meta.names[i], dets[j].prob[i], (left, top, right, bot)))      

    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res

def helmet(im, thresh=0.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    im = im.astype('float32')
    #net = load_net("cfg/yolov3-tiny.cfg".encode("utf-8"), "face_detect_140000.weights".encode("utf-8"), 0)
    #meta = load_meta("cfg/voc.data".encode("utf-8"))
    data = cast(im.ctypes.data, POINTER(c_float))
    im = float_to_image(im.shape[1], im.shape[0], 3, data)
    predict_image(net, im)

    dets = get_network_boxes(net, im.w, im.h,thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]

    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        cla_prob = []
        for ii in range(meta.classes):
            cla_prob.append(dets[j].prob[ii])

        if sum(cla_prob) != 0:
            i = np.argmax(cla_prob)
            print("i",i)
            b = dets[j].bbox
            print("b",b)

            x = b.x
            y = b.y
            w = b.w
            h = b.h

            

            left = int(x - w / 2)
            right = int(x + w / 2)
            top = int(y) - int((h)/2)
            bot = int(y + h/2)

            #if py_version == 3:
                                                                # (  x1,  y1,    x2,  y2)
            #res.append((meta.names[i].decode('utf-8'), dets[j].prob[i], (left, top, right, bot)))
            
            res.append((meta.names[i], dets[j].prob[i], (left, top, right, bot)))      

    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res

class Producer(Process):

    def __init__(self, queue, name, pwd, ip, channel):  # 重写.
        super().__init__()  # 加入父类init.
        self.queue = queue
        self.name = name
        self.pwd = pwd
        self.ip = ip
        self.channel = channel
            
    def run(self):  # call start()时 就会调用run(run为单进程).
        cap = cv2.VideoCapture("rtsp://%s:%s@%s:554/Streaming/Channels/%s" % (self.name, self.pwd, self.ip, self.channel))
        while True:
            is_opened, frame = cap.read()
            self.queue.put(frame) if is_opened else None
            self.queue.get() if self.queue.qsize() > 5 else None


class Consumer(Process):
    def __init__(self, queue, mode, channel):  # 重写.
        super().__init__()  # 加入父类init.
        self.mode = mode
        self.queue = queue
        self.channel = channel

    def run(self):  # call start()时 就会调用run(run为单进程).
        cv2.namedWindow(self.channel, flags=cv2.WINDOW_FREERATIO)
        while True:
            item = self.queue.get()
            if self.mode == 'face':
                result = detect(item)
            elif self.mode == 'helmet':
                result = helmet(item)
            print("result",result)
            cv2.waitKey(1)
            cv2.imshow(self.channel, item)
            self.queue.task_done()


if __name__ == '__main__':
    user_name, user_pwd,camera_ip= "admin", "jdhl123456789", "192.168.1.189"
    channel_l= ["201", "101","301"]
    mp.set_start_method('spawn')

    qMar = Manager()
    # qMar = JoinableQueue()
    queues = [qMar.Queue(3) for _ in channel_l]
    # queue =mp.Queue(maxsize=4)
    
    queue = qMar.Queue(10)

    processes = []
    # processes.append(Producer(queue, user_name, user_pwd, camera_ip, "201"))
    # processes.append(Consumer(queue, 'face', "201", net, meta))
    # p = Producer(queue, user_name, user_pwd, camera_ip, "201")
    # c = Consumer(queue, 'face', "201", 1, 2)
    # p.start()
    # c.start()
    # p.join()
    # c.join()

	# # print(int(cores/2))
    for queue, channel in zip(queues, channel_l):
	    processes.append(Producer(queue, user_name, user_pwd, camera_ip, channel))
	    processes.append(Consumer(queue, 'face', channel))
	    processes.append(Consumer(queue, 'helmet', channel))

		
    [process.start() for process in processes]
    [process.join() for process in processes]

