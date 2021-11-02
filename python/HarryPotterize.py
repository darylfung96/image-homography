import numpy as np
import cv2
import skimage.io 
import skimage.color
import skimage.transform
#Import necessary functions
import random


from planarH import compositeH, computeH_ransac
from loadMatches import loadMatches

###### Section 4.5 

# read the three images
cv_cover = cv2.imread("../data/cv_cover.jpg")
cv_desk = cv2.imread("../data/cv_desk.png")
hp_cover = cv2.imread("../data/hp_cover.jpg")
# read matches of cv_cover and cv_desk
locs1, locs2 = loadMatches("../data/cv_cover.txt", "../data/cv_desk.txt")

# compute the homography using ransac
# np.random.seed(1000)
# random.seed(1000)
H2to1, _ = computeH_ransac(locs1, locs2)

# resize hp_cover to have the same dimension with cv_cover
# you can use opencv function cv2.resize
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

# call compositeH to do the warping
hp_result = compositeH(H2to1, hp_cover, cv_desk)
cv2.imwrite("../result/hp_result.png", hp_result)
