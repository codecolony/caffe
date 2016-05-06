import cv2
# import numpy as np
import imutils
import time

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# if __name__ == "__main__":

def get_windows(fname, image, windowsWidth, windowsHeight):
	# load the image and define the window width and height
	# image = cv2.imread(fname)
	# image = fname
	if image is None:
		print "get_windows(): Image could not be loaded!"
		return ""

	# (winW, winH) = (70, 380)
	(winW, winH) = (windowsWidth, windowsHeight)

	# boxes = u"""filename,xmin,ymin,xmax,ymax
	# group.jpg,42,39,107,317
	# group.jpg,62,69,127,327
	# """

	boxes = ""

	for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			# print "not suitable window"
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW

		boxes = boxes + fname+","+str(x)+","+str(y)+","+str(x+winW)+","+str(y+winH)+"\n"

		# since we do not have a classifier, we'll just draw the window
		# clone = image.copy()
		# cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		# cv2.imshow("Window", clone)
		# cv2.waitKey(1)
		# time.sleep(0.025)

	return boxes

def get_str_to_csv(strName):
	header = "filename,xmin,ymin,xmax,ymax"
	return u""+header+"\n"+strName+"\n"+""