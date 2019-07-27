'''
该文件实现了对数据集的切分，得到总的数据集、训练集、验证集列表txt文件
'''
import os
import cv2
import random

def write_image2txt(images_path, out_path):
    '''
    images_path:图片数据集路径
    out_path:输出文件路径
    '''
    path_list = os.listdir(images_path)
    with open('%s/trainval.txt'%(out_path), 'w+') as out_file:
        for file in path_list:
            out_file.write('%s\n'%(file.split('.')[0]))

def write_image2txt_split(images_path, out_path, ratio):
    '''
    images_path:图片数据集路径
    out_path:输出文件路径
    '''
    path_list = os.listdir(images_path)
    len_list = len(path_list)
    train_list = random.sample(path_list, int(len_list * ratio))
    train_file = open('%s/train.txt'%(out_path), 'w+')
    val_file = open('%s/val.txt'%(out_path), 'w+')
    test_file = open('%s/test.txt'%(out_path), 'w+')
    for file in path_list:
        if file in train_list:
            train_file.write('%s\n'%(file.split('.')[0]))
        else:
            val_file.write('%s\n'%(file.split('.')[0]))
    train_file.close()
    val_file.close()
    test_file.close()


if __name__ == "__main__":
    img_path = 'data/VOCdevkit/VOC2012/JPEGImages'
    set_path = 'data/VOCdevkit/VOC2012/ImageSets/Main'
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    write_image2txt(img_path,set_path)
    write_image2txt_split(img_path,set_path,0.1)
    # os.system("cat VOC_train.txt VOC_val.txt  > train.txt")
    # os.system("cat VOC_train.txt VOC_val.txt VOC_test.txt  > train.all.txt")