# """
# 第一种方式，直接调用opencv中的透视变换函数进行处理
# 进行透视矫正选用的四个点依次为(左上、右上、左下、右下)
# 注意：一定要选准点
# """

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('1.jpg')

# # image = cv2.pyrMeanShiftFiltering(img, 25, 10)
# result = cv2.Canny(imge, 200, 300)
# lines = cv2.HoughLines(edges,1,np.pi/180,160)
# # # 生成透视变换矩阵；进行透视变换
# # M = cv2.getPerspectiveTransform(pts1, pts2)
# # dst = cv2.warpPerspective(img, M, (660,400))
# cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
# # draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3) 
# # draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)  


# cv2.imshow("original_img",img)
# cv2.imshow("result",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

img = cv2.imread('/home/lzc274500/WorkSpace/ZOUZHEN/utils/test_images/11111.jpeg')
img = cv2.pyrDown(img)
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),130, 255,cv2.THRESH_BINARY)  # 黑白二值化
h, w = img.shape[:2]      #获取图像的高和宽 
# cv2.imshow("Origin", img)     #显示原始图像
blured = cv2.GaussianBlur(thresh, (3, 3), 0) 
# blured = cv2.blur(img,(5,5))    #进行滤波去掉噪声
cv2.imshow("Blur", blured)     #显示低通滤波后的图像
 
mask = np.zeros((h+2, w+2), np.uint8)  #掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘 
#进行泛洪填充
cv2.floodFill(blured, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
# cv2.imshow("floodfill", blured) 
 
#得到灰度图
# gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY) 
# cv2.imshow("gray", gray) 
 
 
#定义结构元素 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(70, 70))
#开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
opened = cv2.morphologyEx(blured, cv2.MORPH_OPEN, kernel) 
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) 
cv2.imshow("closed11", closed) 
closed = cv2.Canny(closed,10,50)
cv2.imshow("closed", closed) 

line = 20
minLineLength = 70
maxLineGap = 3
# HoughLinesP函数是概率直线检测，注意区分HoughLines函数
lines = cv2.HoughLinesP(closed, 1, np.pi/180, 60, lines=line, minLineLength=minLineLength,maxLineGap=maxLineGap)
# 降维处理
lines1 = lines[:,0,:]
# line 函数勾画直线
# (x1,y1),(x2,y2)坐标位置
# (0,255,0)设置BGR通道颜色
# 2 是设置颜色粗浅度
for x1,y1,x2,y2 in lines1:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


#找到轮廓
contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
#绘制轮廓
 
cv2.drawContours(img,contours,-1,(0,0,255),3) 
# #绘制结果
cv2.imshow("result", img)
 
cv2.waitKey(0) 
cv2.destroyAllWindows()
# 灰度处理
# grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# blured = cv2.blur(grey,(5,5)) 
# # gradX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# # gradY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
# # gradient = cv2.subtract(gradX, gradY)
# # gradient = cv2.convertScaleAbs(gradient)
# # # canny边缘处理
# edges = cv2.Canny(blured,10,50)
# # line = 20
# # minLineLength = 50
# # maxLineGap = 3
# # # HoughLinesP函数是概率直线检测，注意区分HoughLines函数
# # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, lines=line, minLineLength=minLineLength,maxLineGap=maxLineGap)
# # # 降维处理
# # lines1 = lines[:,0,:]
# # # line 函数勾画直线
# # # (x1,y1),(x2,y2)坐标位置
# # # (0,255,0)设置BGR通道颜色
# # # 2 是设置颜色粗浅度
# # for x1,y1,x2,y2 in lines1:
# #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# # 显示图像
# cv2.imshow("edges", edges)
# cv2.imshow("lines", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
