'''
该文件实现了对错误标注文件，但是有用的部分的摘取
'''
import os
import cv2
import random
import argparse

def write_image2txt_split(label_path, out_path, condition):
    '''
    images_path:图片数据集路径
    out_path:输出文件路径
    '''
    useful_file = open('%s/useful.txt'%(out_path), 'w+')
    res_file = open('%s/res.txt'%(out_path), 'w+')

    with open(label_path, "r") as label_data:
        for i in label_data:  # 逐行读取
            tmp = i.split()
            # print(picture_name.split('.')[0],tmp[0])
            if tmp[1] == condition:
                res_file.write(i)
            else:
                useful_file.write(i)
    useful_file.close()
    res_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default=None, help='condition of filter')
    parser.add_argument('--filepath', type=str, default=None, help='path of file')
    parser.add_argument('--savepath', type=str, default=None, help='path of save')

    opt = parser.parse_args()

    file_path = '/home/lzc274500/WorkSpace/ZOUZHEN/datasets/8-20/返工总.txt'
    set_path = '/home/lzc274500/WorkSpace/ZOUZHEN/datasets/8-20'
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    
    write_image2txt_split(opt.filepath,opt.savepath,opt.condition)
