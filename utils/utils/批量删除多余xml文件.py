import os

'''
本脚本实现了对不同文件的脚本
'''

@staticmethod
def funcname(parameter_list):
     pass

files = os.listdir("/home/jdhl/darknet/scripts/VOCdevkit/VOC2007/JPEGImages")#获取当前目录下的文件
# filesb = os.listdir("/home/jdhl/darknet/scripts/VOCdevkit/VOC2007/result")#获取当前目录下的文件
# files = files + filesb
files1 = os.listdir("/home/jdhl/darknet/scripts/VOCdevkit/VOC2007/Annotations")#获取当前目录下的文件
# files1 = os.listdir("/home/jdhl/WorkSpace/ZOUZHEN/dataset/7-02/仪表标注数据/c_xml")#获取当前目录下的文件
list0 = []
#print(files1)
for i in files:
    filename = os.path.splitext(i)[0]#将文件名拆成名字和后缀
    file_name=filename + ".xml"#把后缀改为xml
    list0.append(file_name) #在list0列表中逐个添加
# list_merge=list(set(list0).difference(set(files1))) #求两列表差集
list_merge=list(set(files1).difference(set(list0))) #求两列表差集
# list_merge=list(set(list0).intersection(set(files1)) ) #求两列表交集
# list_merge=list(set(list0).difference(set(list_merge)) ) #求两列表交集
# list_merge=list(set(list0).difference(set(list_merge))) #求两列表并集
#list_merge=list(set(list0)&(set(files1))) 
print('元数据',len(list0))         
print('已合并的数据',len(files1))
print('交集',len(list_merge))
# print(list_merge[0])
# print(os.path.splitext(list_merge[0])[0])
# for j in range(len(files)):
#      os.remove(r'/media/jdhl/Elements/本安安全帽数据集/Annotations/'+os.path.splitext(list_merge[j])[0]+'.xml')  #####删除xml文件
#      # os.remove(r'/media/jdhl/Elements/本安安全帽数据集/JPEGImages/'+list_merge[j])  #####删除xml文件
# print(list0[0])
for file in list_merge:
     os.remove(r'/home/jdhl/darknet/scripts/VOCdevkit/VOC2007/Annotations/'+file)  #####删除xml文件
     # os.remove(r'/media/jdhl/Elements/本安安全帽数据集/JPEGImages/'+list_merge[j])  #####删除xml文件