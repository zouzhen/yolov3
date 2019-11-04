import os.path
import cv2
import pandas as pd

from datetime import datetime

# 数据集的路径（PS：该文件夹中只能放图片）
file = "/home/jdhl/WorkSpace/ZOUZHEN/dataset/8-9/xa变电站识别"

# 添加标签后的数据路径
save_file="/home/jdhl/WorkSpace/ZOUZHEN/dataset/8-9/result"
# save_files="/home/jdhl/WorkSpace/ZOUZHEN/dataset/7-02/仪表标注数据/error_image"
if not os.path.exists(save_file):
    os.makedirs(save_file)
files = os.listdir(save_file)
# 标注文件路径
label_path = "/home/jdhl/WorkSpace/ZOUZHEN/dataset/8-8/useful.txt"
data = pd.read_csv(label_path,sep=' ',header=None)
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

for picture_name in result:
    file_img = cv2.imread(os.path.join(file, picture_name))
    print(picture_name)
    with open(label_path, "r") as label_data:
        for i in label_data:  # 逐行读取
            tmp = i.split()
            # print(picture_name.split('.')[0],tmp[0])
            if picture_name == tmp[0]:
                # print("tmp[0]",tmp[0])
                # print("tmp[0]",tmp[0])
                # print("tmp[1]",tmp[2:])
                boxes=tmp[2:]

                for i in range(len(boxes)):
                    xmin = int(boxes[0])
                    ymin = int(boxes[1])
                    xmax = int(boxes[2])
                    ymax = int(boxes[3])
                    cv2.putText(file_img, tmp[1], (xmin,ymin), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,255,0), 1)


    cv2.imwrite(save_file +'/'+ picture_name,file_img)

# # for picture_name in files:
# #     file_img = cv2.imread(os.path.join(file, picture_name))
# #     find_data = data[data[0]==picture_name].as_matrix()
# #     for tmp in find_data:  # 逐行读取
# #         # print("tmp[0]",tmp[0])
# #         # print("tmp[0]",tmp[0])
# #         # print("tmp[1]",tmp[2:])
# #         boxes=tmp[2:]

# #         for i in range(len(boxes)):
# #             xmin = int(boxes[0])
# #             ymin = int(boxes[1])
# #             xmax = int(boxes[2])
# #             ymax = int(boxes[3])
# #             if tmp[1]== 'Standard':
# #                 cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,255,0), 1)
# #             else:
# #                 cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,0,255), 1)

# #     cv2.imwrite(save_file +'/'+ picture_name,file_img)

end = datetime.now()

print("耗时：", end-start)

# if __name__ == "__main__":
#     relabel()
