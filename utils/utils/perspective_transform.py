"""
第一种方式，直接调用opencv中的透视变换函数进行处理
进行透视矫正选用的四个点依次为(左上、右上、左下、右下)
注意：一定要选准点
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('1.jpg')
H_rows, W_cols= img.shape[:2]
print(H_rows, W_cols)
W_cols,H_rows = 660,400
print(H_rows, W_cols)

# 原图中书本的四个角点(左上、右上、右下、左下),与变换后矩阵位置
pts1 = np.float32([[80, 640],  [1180, 640], [1050, 115],[200, 125]])
pts2 = np.float32([[0, H_rows],[W_cols,H_rows],[W_cols,0],[0, 0]])


# 生成透视变换矩阵；进行透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (660,400))

cv2.imshow("original_img",img)
cv2.imshow("result",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
第二种方式，直接编写透视变换的代码
代码在这个链接（https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/）
进行透视矫正选用的四个点依次为(左上、右上、左下、右下)
注意：一定要选准点
执行脚本: python perspective_transform.py --image 1.jpg --coords "[(80, 640), (1180, 640), (1050, 115),(200, 125)]"
"""

# import the necessary packages
import numpy as np
import argparse
import cv2

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    ap.add_argument("-c", "--coords",
        help = "comma seperated list of source points")
    args = vars(ap.parse_args())
    # load the image and grab the source coordinates (i.e. the list of
    # of (x, y) points)
    # NOTE: using the 'eval' function is bad form, but for this example
    # let's just roll with it -- in future posts I'll show you how to
    # automatically determine the coordinates without pre-supplying them
    image = cv2.imread(args["image"])
    pts = np.array(eval(args["coords"]), dtype = "float32")
    
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(image, pts)
    
    # show the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)