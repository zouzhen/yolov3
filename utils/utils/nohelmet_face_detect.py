# coding:utf-8
import sys
from ctypes import *
import random
import cv2
import copy
import numpy as np
import time
import face_model
import argparse
import dlib
import os
import math
import requests
import multiprocessing as mp

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


lib = CDLL("libdarknet.so", RTLD_GLOBAL)
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

def upload_event_message(url, camera_ip, event_type, timestamp, photo_information):
    requests.post(url, data={'camera_ip':camera_ip,'event_type':event_type,'timestamp':timestamp,'photo_information':photo_information})
    return True

# use cv2.imread; one box predict one class
def detect(im, thresh=0.5, hier_thresh=.5, nms=.45):
    #import time
    #start_time = time.time()
    #im = cv2.imread(image)
    num = c_int(0)
    pnum = pointer(num)
    im = im.astype('float32')

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
            #print("i",i)
            b = dets[j].bbox
            #print("b",b)

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


def detect_face(im, thresh=0.5, hier_thresh=.5, nms=.45):
    # import time
    # start_time = time.time()
    # im = cv2.imread(image)
    num = c_int(0)
    pnum = pointer(num)
    im = im.astype('float32')

    data = cast(im.ctypes.data, POINTER(c_float))
    im = float_to_image(im.shape[1], im.shape[0], 3, data)
    predict_image(net_face, im)

    dets = get_network_boxes(net_face, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]

    if (nms): do_nms_obj(dets, num, meta_face.classes, nms);

    res = []
    for j in range(num):
        cla_prob = []
        for ii in range(meta_face.classes):
            cla_prob.append(dets[j].prob[ii])

        if sum(cla_prob) != 0:
            i = np.argmax(cla_prob)
            # print("i",i)
            b = dets[j].bbox
            # print("b",b)

            x = b.x
            y = b.y
            w = b.w
            h = b.h

            left = int(x - w / 2)
            right = int(x + w / 2)
            top = int(y) - int((h) / 2)
            bot = int(y + h / 2)

            # if py_version == 3:
            # (  x1,  y1,    x2,  y2)
            # res.append((meta.names[i].decode('utf-8'), dets[j].prob[i], (left, top, right, bot)))

            res.append((meta_face.names[i], dets[j].prob[i], (left, top, right, bot)))

    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res


def video_test_helmets(video):

    cap = cv2.VideoCapture(video)

    while (True):
        ret, frame = cap.read()

        result = detect(frame)

        for res in result:
            cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]
            cv2.putText(frame, str(cls) + ": " + str(prob)[:5], (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
            print("result", result)
            cv2.imshow("Demo", frame)
            cv2.waitKey(1)


def pic_test_defence(pic):
    img = cv2.imread(pic)
    result = detect(img)
    for res in result:
            cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]
            #cv2.putText(img, str(cls) + ": " + str(prob)[:5], (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Demo", img)
    cv2.waitKey(0)




def pic_helmet_face_detect(pic):
    img = cv2.imread(pic)
    result = detect(img)
    # print("result", result)
    for res in result:
        cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]
        # cv2.putText(img, str(cls) + ": " + str(prob)[:5], (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        # print("cls", cls)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        if cls == "noHelmet":
            dec_people = img[y1:y2, x1:x2]
            result_face = detect_face(dec_people)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            print("result_face", result_face)
            for res_face in result_face:
                cls, prob, x1, y1, x2, y2 = res_face[0], res_face[1], res_face[2][0], res_face[2][1], res_face[2][
                    2], res_face[2][3]
                # cv2.putText(dec_people, str(cls) + ": " + str(prob)[:5], (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
                # print("result", result)
                cv2.rectangle(dec_people, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
                print("cls", cls)


            # cv2.imshow('image', dec_people)
            # cv2.waitKey(0)

    cv2.imshow("Demo", img)
    cv2.waitKey(0)


def pic_helmet_face_detect1(pic):
    img = cv2.imread(pic)
    result = detect(img)
    #print("result", result)
    for res in result:
            cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]
            #cv2.putText(img, str(cls) + ": " + str(prob)[:5], (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            dec_people = img[y1:y2, x1:x2]
            result_face = detect_face(dec_people)
            print("result_face", result_face)
            for res_face in result_face:
                cls, prob, x1, y1, x2, y2 = res_face[0], res_face[1], res_face[2][0], res_face[2][1], res_face[2][2], res_face[2][3]
                #cv2.putText(dec_people, str(cls) + ": " + str(prob)[:5], (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
                #print("result", result)
                cv2.rectangle(dec_people, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

            # cv2.imshow('image', dec_people)
            # cv2.waitKey(0)

    cv2.imshow("Demo", img)
    cv2.waitKey(0)




def face_detecter_align1(img):#sample face align
    result = detect_face(img)

    if len(result) == 0:   #判断是否有人脸,没有返回None
        return None, None
    else:

        res = result[0][2]
        x1, y1, x2, y2 = res[0], res[1], res[2], res[3]
        x1 = max(x1-20, 0)
        y1 = max(y1+20, 0)
        x2 = max(x2-20, 0)
        y2 = max(y2+20, 0)
        print("y2",y2)
        bboxs = (x1, y1, x2, y2)
        dec_face = img[y1:y2, x1:x2, :]
        dec_face = cv2.resize(dec_face, (112, 112), interpolation=cv2.INTER_AREA)
            # print("dec_face", dec_face.shape)
        gray1 = cv2.cvtColor(dec_face, cv2.COLOR_BGR2RGB)
        dets = detector(gray1, 1)
        if len(dets) != 0:   #如果有人脸返回对其人脸和坐标
            aligned1 = np.zeros((3,112, 112), dtype=np.int)

            for face in dets:
                print("face", face)
                shape = predictor(gray1, face)  # 寻找人脸的68个标定点
                eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼中心坐标
                              (shape.part(36).y + shape.part(45).y) * 1. / 2)
                # print("eye_center",eye_center)
                dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
                dy = (shape.part(45).y - shape.part(36).y)

                angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度

                RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 仿射矩阵

                RotImg = cv2.warpAffine(dec_face, RotateMatrix, (dec_face.shape[1], dec_face.shape[0]))

                if dec_face.shape[1] >= 112:
                    RotImg = cv2.resize(RotImg, (112, 112), interpolation=cv2.INTER_CUBIC)
                else:
                    RotImg = cv2.resize(RotImg, (112, 112), interpolation=cv2.INTER_AREA)

                alignedd = np.transpose(RotImg, (2, 0, 1))

                aligned1 = np.concatenate(( aligned1,alignedd), axis=0)

            aligned2 = aligned1[3:,:,:]

            aligned2 = np.transpose(aligned2, (1, 2, 0))

            return aligned2,bboxs
        else:
            print("face0000")
            return None, None   #二次判断是否有人脸,没有返回None

def face_detecter_align(img):#sample face align
    result = detect_face(img)

    if len(result) == 0:   #判断是否有人脸,没有返回None
        return None, None
    else:

        res = result[0][2]
        x1, y1, x2, y2 = res[0], res[1], res[2], res[3]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = max(x2, 0)
        y2 = max(y2, 0)
        print("y2",y2)
        bboxs = (x1, y1, x2, y2)
        dec_face = img[y1:y2, x1:x2, :]
        gray1 = cv2.cvtColor(dec_face, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        shape = predictor(gray1, rect) # 寻找人脸的68个标定点
        if shape is None:
            return None, None
        else:
            eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼中心坐标
                          (shape.part(36).y + shape.part(45).y) * 1. / 2)
            print('eye_center',eye_center)

            dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
            dy = (shape.part(45).y - shape.part(36).y)

            angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度

            RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 仿射矩阵

            #RotImg = cv2.warpAffine(dec_face, RotateMatrix, (dec_face.shape[1], dec_face.shape[0]))

            RotImg = cv2.warpAffine(img, RotateMatrix, (dec_face.shape[1], dec_face.shape[0]))
            
            if dec_face.shape[1] >= 112:
                RotImg = cv2.resize(RotImg, (112, 112), interpolation=cv2.INTER_CUBIC)
            else:
                RotImg = cv2.resize(RotImg, (112, 112), interpolation=cv2.INTER_AREA)
            cv2.imshow("Camera", RotImg)
            cv2.waitKey(0)
            aligned = np.transpose(RotImg, (2, 0, 1))
            #aligned = cv2.cvtColor(RotImg, cv2.COLOR_RGB2BGR)
            return aligned , bboxs


def test_Faces(args):

    model = face_model.FaceModel(args)
    imgs = os.listdir(args.image_path)
    # 设置阈值，这两个阈值用来判定该人是否是库里的人
    a = zip([args.threshold1], [args.threshold2])
    for k, v in a:
        for img in imgs:
            ipic = cv2.imread(os.path.join(args.image_path, img))

            pic, bboxs = face_detecter_align(ipic)
            #print("pic",pic)

            #print("pic.len", len(pic))
            #start = time.time()
            if pic is None:
                continue
            else:
                draw = ipic.copy()
                for m in range(1, (pic.shape[0] / 3)+1):


                    apic = pic[(3*m-3):m*3]
                    #end = time.time()
                    abbox = bboxs[4*m-4:(4*m)]
                    #apoints = points[10*m-10:10*m]
                    cv2.rectangle(draw, (int(abbox[0]), int(abbox[1])), (int(abbox[2]), int(abbox[3])), (0, 255, 0), 2)
                    # for l in range(0, 5):
                    #     cv2.circle(draw, (int(apoints[l]), int(apoints[l + 5])), 2, (0, 255, 0), 2)

                    f1 = model.get_feature(apic)
                    ############################
                    maxl = []
                    for i in range(faces.shape[0]):
                        cnt = 0
                        for j in range(faces.shape[1]):
                            dist = np.sqrt(np.sum(np.square(f1 - faces[i][j])))
                            # 如果与库中某人距离小于阈值1
                            if dist < k:
                                cnt += 1
                        z = [i, cnt]
                        maxl = maxl + z
                    e = maxl[1::2].index(max(maxl[1::2]))
                    # 如果与库中某人距离小于阈值2
                    if max(maxl[1::2]) >= v:
                        name = labels[e]
                        testresult.writelines(os.path.join(args.image_path, img) + ' is ' + name + '\n')

                    else:
                        testresult.writelines(os.path.join(args.image_path, img) + ' is ' + "unknow" + '\n')
                        name = "unknow"
                    cv2.putText(draw, name, (int(abbox[0]), int(abbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Camera", draw)
            cv2.imwrite('chip_' + str(img) + '.png', draw)
    testresult.close()
    #print('Running time: {} Seconds'.format(end-start))



def video_test_Faces(args):
    start = time.time()

    model = face_model.FaceModel(args)
   # 设置阈值，这两个阈值用来判定该人是否是库里的人
    a = zip([args.threshold1], [args.threshold2])
    for k, v in a:
        cap = cv2.VideoCapture("/home/lzc274500/nohelmet_face_detect/02.mp4")
        while (True):
            ret, frame = cap.read()
            pic, bboxs = face_detecter_align(frame)

            if pic is None:  #判断是否有人脸,没有人脸返回
                continue
            else:
                for m in range(1, (pic.shape[0] / 3)+1):
                    # print("m", m)
                    apic = pic[(3*m-3):m*3]
                    abbox = bboxs[4*m-4:(4*m)]
                    #apoints = points[10*m-10:10*m]
                    cv2.rectangle(frame, (int(abbox[0]), int(abbox[1])), (int(abbox[2]), int(abbox[3])), (0, 255, 0), 2)
                    # for l in range(0, 5):
                    #     cv2.circle(frame, (int(apoints[l]), int(apoints[l + 5])), 2, (0, 255, 0), 2)
                    f1 = model.get_feature(apic)
                    maxl = []
                    for i in range(faces.shape[0]):
                        cnt = 0
                        for j in range(faces.shape[1]):
                            dist = np.sqrt(np.sum(np.square(f1 - faces[i][j])))
                            # 如果与库中某人距离小于阈值1
                            if dist < k:
                                cnt += 1
                        z = [i, cnt]
                        maxl = maxl + z
                    e = maxl[1::2].index(max(maxl[1::2]))
                    # 如果与库中某人距离小于阈值2
                    if max(maxl[1::2]) >= v:
                        name = labels[e]
                    else:
                        name = "unknow"
                    cv2.putText(frame, name, (int(abbox[0]), int(abbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #out.write(frame)
            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()  # 关闭相机
        #out.release()
        cv2.destroyAllWindows()
            #cv2.imwrite('chip_' + str(img) + '.png', draw)
    testresult.close()
    end = time.time()
    print('Running time: {} Seconds'.format(end-start))


def nohelmet_face_detect(args):
    k= 1.37
    v= 1.8
    model = face_model.FaceModel(args)
    img = cv2.imread(pic_pic)
    result = detect(img)
    # print("result", result)
    for res in result:
        cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]

        if cls == "withHelmet": #没有戴安全帽的进行人脸识别,并画红框
            dec_people = img[y1:y2, x1:x2]


            cv2.rectangle(img, (x1, y1), (x2,y2), (0, 0, 255), 2)# 不带安全帽画红框

            picc, bboxs =face_detecter_align(dec_people)
            #print("pic",picc.shape)
            #print("bboxs", bboxs)

            if picc is None:  #如果有人脸,进行下一步身份判定
                # 计算出face在整体画面的实际坐标,并标出
                name = "unknow"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 不带安全帽画红框
                cv2.putText(img, name, (x1, int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                print("name2", name)

            else:   #如果没有人脸,身份识别终止,名称返回空

                f_x1 = int(x1 + bboxs[0])
                f_y1 = int(y1 + bboxs[1])
                f_x2 = int(x1 + bboxs[2])
                f_y2 = int(y1 + bboxs[3])
                print("name1", f_x1)
                #cv2.rectangle(img, (f_x1, f_y1), (f_x2, f_y2), (0, 0, 255), 2)
                print("name3", f_y1)
                f1 = model.get_feature(picc)
                print("name4", f_x2)
                print("name5", f_y2)

                maxl = []
                for i in range(faces.shape[0]):
                    cnt = 0
                    for j in range(faces.shape[1]):
                        dist = np.sqrt(np.sum(np.square(f1 - faces[i][j])))
                        print("dist", dist)
                        # 如果与库中某人距离小于阈值1
                        if dist < k:
                            cnt += 1
                    z = [i, cnt]
                    maxl = maxl + z
                e = maxl[1::2].index(max(maxl[1::2]))
                # 如果与库中某人距离小于阈值2
                if max(maxl[1::2]) >= v:
                    name = labels[e]
                else:
                    name = "unknow"
                cv2.putText(img, name, (f_x1, int(f_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                #print("name1", name)
                cv2.rectangle(img, (f_x1, f_y1), (f_x2, f_y2), (0, 0, 255), 2)  # 不带安全帽画红框

        else:   #戴安全帽画个绿框
            print("name31111")
            cv2.rectangle(img, ((x1, y1)), (x2, y2), (0, 255,0), 2)  # 带安全帽画绿框

    cv2.imshow("Demo", img)
    cv2.waitKey(0)



def queue_img_face_helmet(args,q, window_name):#withhelmet and nohelmet detect face and draw rectangle
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    k= 1.37
    v= 1.8
    model = face_model.FaceModel(args)
    while True:
        img = q.get()
        result = detect(img)
        # print("result", result)
        for res in result:
            cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]

            if cls == "withHelmet": #没有戴安全帽的进行人脸识别,并画红框
                upload_event_message(url,camera_ip,cls,timestamp,photo_information)
                dec_people = img[y1:y2, x1:x2]


                cv2.rectangle(img, (x1, y1), (x2,y2), (0, 0, 255), 2)# 不带安全帽画红框

                picc, bboxs =face_detecter_align(dec_people)
                #print("pic",picc.shape)
                #print("bboxs", bboxs)

                if picc is None:  #如果有人脸,进行下一步身份判定
                    # 计算出face在整体画面的实际坐标,并标出
                    name = "unknow"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 不带安全帽画红框
                    cv2.putText(img, name, (x1, int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    print("name2", name)

                else:   #如果没有人脸,身份识别终止,名称返回空

                    f_x1 = int(x1 + bboxs[0])
                    f_y1 = int(y1 + bboxs[1])
                    f_x2 = int(x1 + bboxs[2])
                    f_y2 = int(y1 + bboxs[3])
                    print("name1", f_x1)
                    #cv2.rectangle(img, (f_x1, f_y1), (f_x2, f_y2), (0, 0, 255), 2)
                    print("name3", f_y1)
                    f1 = model.get_feature(picc)
                    print("name4", f_x2)
                    print("name5", f_y2)

                    maxl = []
                    for i in range(faces.shape[0]):
                        cnt = 0
                        for j in range(faces.shape[1]):
                            dist = np.sqrt(np.sum(np.square(f1 - faces[i][j])))
                            print("dist", dist)
                            # 如果与库中某人距离小于阈值1
                            if dist < k:
                                cnt += 1
                        z = [i, cnt]
                        maxl = maxl + z
                    e = maxl[1::2].index(max(maxl[1::2]))
                    # 如果与库中某人距离小于阈值2
                    if max(maxl[1::2]) >= v:
                        name = labels[e]
                    else:
                        name = "unknow"
                    cv2.putText(img, name, (f_x1, int(f_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    #print("name1", name)
                    cv2.rectangle(img, (f_x1, f_y1), (f_x2, f_y2), (0, 0, 255), 2)  # 不带安全帽画红框

            else:   #戴安全帽画个绿框
                print("name31111")
                cv2.rectangle(img, ((x1, y1)), (x2, y2), (0, 255,0), 2)  # 带安全帽画绿框

        cv2.imshow("Demo", img)
        cv2.waitKey(0)

def queue_img_put(q, name, pwd, ip, channel):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s:554/Streaming/Channels/%s" % (name, pwd, ip, channel))
    #cap = cv2.VideoCapture("/home/lzc274500/face_detect_yolov3-init/suzhen.mp4" )
    #cap = cv2.VideoCapture("rtsp://admin:jdhl123456789@192.168.1.189:554/Streaming/Channels/101")
    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 5 else None

def queue_img_face(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        #print("frame", frame)
        result = detect(frame)
        print("result",result)
        cv2.waitKey(1)
        cv2.imshow(window_name, frame)


def run_multi_camera(arg):
    user_name, user_pwd,camera_ip= "admin", "jdhl123456789", "192.168.1.189"
    #channel_l = [
    #     "201",
    #     "101",
    # ]

    channel_l= [
        "201", "101","301"]
    mp.set_start_method('spawn')

    queues = [mp.Queue(maxsize=6) for _ in channel_l]

    processes = []
    for queue, channel in zip(queues, channel_l):
        processes.append(mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip, channel)))
        processes.append(mp.Process(target=queue_img_face_helmet, args=(arg, queue, channel)))

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]

if __name__ == '__main__':
    #import glob, os
    net = load_net("/home/lzc274500/defence_detect/cfg/defence_detect.cfg", "/home/lzc274500/defence_detect/defence_detect.weights", 0)

    net_face = load_net("/home/lzc274500/nohelmet_face_detect/cfg/face_detect.cfg", "/home/lzc274500/nohelmet_face_detect/face_detect_140000.weights", 0)

    meta = load_meta("/home/lzc274500/defence_detect/cfg/voc.data")

    meta_face = load_meta("/home/lzc274500/nohelmet_face_detect/cfg/face_detect.data")

    #video = "/home/lzc274500/defence_detect/video_003.dav"

    #pic_pic = "/home/lzc274500/nohelmet_face_detect/0203625.jpg"

    #pic_pic =  "/home/lzc274500/nohelmet_face_detect/0203865.jpg"

    #pic_pic = "/home/lzc274500/nohelmet_face_detect/0202161.jpg"
    pic_pic = "/home/lzc274500/nohelmet_face_detect/0101897.jpg"

    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model',
                        default='/home/lzc274500/insightface-master/deploy/models/r100-arcface-emore/model,0001',
                        help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--image_path', default='/home/lzc274500/insightface-master/deploy/tests',
                        help='test image path')
    # parser.add_argument('--image_path', default='/media/lzc274500/Elements/safety1', help='test image path')

    parser.add_argument('--threshold', default=1.37, type=float, help='ver dist threshold')
    parser.add_argument('--threshold2', default=18, type=int, help='threshold for those whose dist is above threshold1')
    parser.add_argument('--threshold1', default=1.37, type=float,
                        help='threshold for those whose dist is above threshold1')
    args = parser.parse_args()
    face_path = '/home/lzc274500/insightface-master/deploy/gallery/company_gallery/face_gallery.npy'
    label_path = '/home/lzc274500/insightface-master/deploy/gallery/company_gallery/face_labels.npy'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/lzc274500/face_detect_yolov3-init/shape_predictor_68_face_landmarks.dat")
    faces = np.load(face_path)
    labels = np.load(label_path)


    #video_test_Faces(args)
    #nohelmet_face_detect(args)
    run_multi_camera(args)
    #pic_helmet_face_detect(pic)
    #face_detecter_align(pic_pic)



