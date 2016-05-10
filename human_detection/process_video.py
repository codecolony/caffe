import cv2
import imutils
import sys
import caffe_detect_module

frame_scale = 5

cap = cv2.VideoCapture("team.mov")

w  = int(cap.get(3))
h = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'xvid')
vw = cv2.VideoWriter("output.avi",fourcc, 20.0, (w/frame_scale, h/frame_scale))

if cap is None:
	print "process_video(): Error opening video file."
	sys.exit(0)

if not vw.isOpened():
	print "Error opening video writer!"
	sys.exit(0)

k = 0
fname = "human.jpg"
frame_count = 0
while(1):

	# if frame_count == 5:
	# 	print "configured number of frames written to video."
	# 	break

	(grabbed, frame) = cap.read()

	if not grabbed:
		print "End of video stream reached."
		break

	if k % 5 != 1:
		k = k + 1
		continue

	k = k + 1

	frame = imutils.resize(frame, width=w/frame_scale)
	cv2.imwrite("human.jpg", frame)
	boxes = caffe_detect_module.get_caffe_detections(fname, frame)

	for i in boxes:
		cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)

	vw.write(frame)
	print "frame "+ str(k)+ " written to file"
	frame_count = frame_count + 1

cap.release()
vw.release()
cv2.destroyAllWindows()