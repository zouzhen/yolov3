'''
该文件实现了对错误标注文件，但是有用的部分的摘取
'''
import os
import cv2
import random


def write_image2txt_split(label_path, out_path, contition='RotarySwitchOFF'):
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
            if tmp[1] == contition:
                res_file.write(i)
            else:
                useful_file.write(i)
    useful_file.close()
    res_file.close()



if __name__ == "__main__":
    file_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/8-8/a1318.txt'
    set_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/8-8'
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    write_image2txt_split(file_path,set_path)
