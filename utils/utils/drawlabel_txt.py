import os.path
import cv2
import pandas as pd
import os 
from datetime import datetime

# 数据集的路径（PS：该文件夹中只能放图片）
fileA = "/home/lzc274500/WorkSpace/ZOUZHEN/datasets/8-26/Untitled Folder 3"
file = os.listdir(fileA)
# 添加标签后的数据路径
save_file="/home/lzc274500/WorkSpace/ZOUZHEN/datasets/8-26/result.txt"

# 标注文件路径
label_path = "/home/lzc274500/WorkSpace/ZOUZHEN/datasets/8-26/xml/1111111111.txt"
print('============================')
data = pd.read_csv(label_path,sep=' ',header=None,error_bad_lines=False)
print('============================')
result = data.drop_duplicates([0])[0].values

def find_diff(fileA,fileB,suffix):
    """
    the files of filesA is more than filesB 
    """
    filesA = os.listdir(fileA)
    filesB = os.listdir(fileB)
    list0 = []
    for i in filesB:
        filename = os.path.splitext(i)[0]#将文件名拆成名字和后缀
        file_name=filename + suffix#把后缀改为xml
        list0.append(file_name) #在list0列表中逐个添加
    list_merge=list(set(filesA).difference(set(filesB)))
    # list_merge=list(set(list0).intersection(set(files1)) ) #求两列表并集
    # list_merge=list(set(list0).difference(set(list_merge))) #求两列表并集
    print('交集',len(list_merge))
    return list_merge

# result = find_diff(file,save_file,'jpg')

start = datetime.now()

label_txt = open(save_file,'a+')

for picture_name in file:
    print(picture_name)
    # print(os.path.join(file, picture_name))
    with open(label_path, "r") as label_data:
        for i in label_data:  # 逐行读取
            tmp = i.split()
            label_txt.write(picture_name+' '+str(tmp[1])+' '+str(tmp[2])+' '+str(tmp[3])+' '+str(tmp[4])+' '+str(tmp[5])+os.linesep)


end = datetime.now()

print("耗时：", end-start)

# if __name__ == "__main__":
#     relabel()
