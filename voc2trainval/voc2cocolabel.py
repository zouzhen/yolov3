import pickle
import os
from os.path import join
from os import listdir, getcwd
import xml.etree.ElementTree as ET


classes = ["BigShavings", "GlueSpot", "OilPollution",
           "SandLeakage", "Soft", "Crack", "EdgeCollapse"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def convert_annotation(input_path, image_id):
    in_file = open(os.path.join(input_path, 'annotations/'+image_id+'.xml'))
    out_file = open(os.path.join(input_path, 'labels/'+image_id+'.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    input_path = input()
    if not os.path.exists('%s' % (os.path.join(input_path, 'labels'))):
        os.makedirs(os.path.join(input_path, 'labels'))
    image_ids = os.listdir(os.path.join(input_path, 'images'))
    for image_id in image_ids:
        convert_annotation(input_path, os.path.splitext(image_id)[0])
