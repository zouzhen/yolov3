import os.path
import cv2
import pandas as pd

from datetime import datetime

# 数据集的路径（PS：该文件夹中只能放图片）
file = "/home/jdhl/WorkSpace/ZOUZHEN/dataset/100张分两个50张做的/劉剛"

# 添加标签后的数据路径
save_file="/home/jdhl/WorkSpace/ZOUZHEN/dataset/100张分两个50张做的/刘刚标注"

files = os.listdir(file)
# 标注文件路径
label_path = "/home/jdhl/WorkSpace/ZOUZHEN/dataset/100张分两个50张做的/刘钢5月31.txt"
data = pd.read_csv(label_path,sep=' ',header=None)

start = datetime.now()

for picture_name in files:
    file_img = cv2.imread(os.path.join(file, picture_name))
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
                    if tmp[1]== 'withHelmet':
                        cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,255,0), 1)
                    else:
                        cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,0,255), 1)

    cv2.imwrite(save_file +'/'+ picture_name,file_img)

# for picture_name in files:
#     file_img = cv2.imread(os.path.join(file, picture_name))
#     find_data = data[data[0]==picture_name].as_matrix()
#     for tmp in find_data:  # 逐行读取
#         # print("tmp[0]",tmp[0])
#         # print("tmp[0]",tmp[0])
#         # print("tmp[1]",tmp[2:])
#         boxes=tmp[2:]

#         for i in range(len(boxes)):
#             xmin = int(boxes[0])
#             ymin = int(boxes[1])
#             xmax = int(boxes[2])
#             ymax = int(boxes[3])
#             if tmp[1]== 'withHelmet':
#                 cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,255,0), 1)
#             else:
#                 cv2.rectangle(file_img,(xmin,ymin),(xmax,ymax), (0,0,255), 1)

#     cv2.imwrite(save_file +'/'+ picture_name,file_img)

end = datetime.now()

print("耗时：", end-start)

# if __name__ == "__main__":
#     relabel()
