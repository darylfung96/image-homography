import numpy as np
import cv2
from helper import plotMatches

def loadMatches(f1, f2):
	# read from two txt files
	with open(f1, 'r') as f:
		f1_coords = f.read()
		locs1 = np.array([item.split() for item in f1_coords.split('\n')[:-1]]).astype(np.int32)
	with open(f2, 'r') as f:
		f2_coords = f.read()
		locs2 = np.array([item.split() for item in f2_coords.split('\n')[:-1]]).astype(np.int32)
	
	return locs1, locs2


if __name__ == "__main__":
	cv_cover = cv2.imread('../data/cv_cover.jpg')
	cv_desk = cv2.imread('../data/cv_desk.png')
	locs1, locs2 = loadMatches('../data/cv_cover.txt', '../data/cv_desk.txt')

	plotMatches(cv_cover, cv_desk, locs1, locs2)
