import os
import xml.etree.ElementTree as xmlET
files = os.listdir("/home/jdhl/WorkSpace/ZOUZHEN/dataset/6-28/123")#获取当前目录下的文件
# filesb = os.listdir("/home/jdhl/darknet/scripts/VOCdevkit/VOC2007/result")#获取当前目录下的文件
# files = files + filesb
files1 = "/home/jdhl/WorkSpace/ZOUZHEN/dataset/6-28/本安安全帽数据集/Annotations_correction" #获取当前目录下的文件
save_path = '/home/jdhl/WorkSpace/ZOUZHEN/dataset/6-28/changed_xml'
list0 = []
#print(files1)
for i in files:
    filename = os.path.splitext(i)[0]#将文件名拆成名字和后缀
    file_name=filename + ".xml"#把后缀改为xml
    list0.append(file_name) #在list0列表中逐个添加  
print(len(list0))
for xml in list0:
    print(xml)
    tree = xmlET.parse(os.path.join(files1,xml))
    root = tree.getroot()

    for obj in root.findall('object'):
        obj.find('name').text = 'withHelmet'
    
    tree.write(os.path.join(save_path, xml))
