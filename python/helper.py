import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.feature

def plotMatches(im1,im2,locs1,locs2):
	fig, ax = plt.subplots(nrows=1, ncols=1)
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	matches = np.array([np.arange(locs1.shape[0]), np.arange(locs1.shape[0])]).T
	plt.axis('off')
	skimage.feature.plot_matches(ax,im1,im2,locs1[:,[1, 0]],locs2[:,[1, 0]],matches,keypoints_color='r')
	plt.show()
	return