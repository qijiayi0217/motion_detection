# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=600, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
count=1
preframe=[]
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"
	#print(type(preframe))
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	#print(type(preframe))
	# if the first frame is None, initialize it
	# if firstFrame is None:
	# 	firstFrame = gray
	# 	continue
	if len(preframe)!=5:
		preframe.append(gray)
		continue
	else:
		#firstFrame=preframe[0]
		preframe[0]=preframe[1]
		preframe[1]=preframe[2]
		preframe[2]=preframe[3]
		preframe[3]=preframe[4]
		preframe[4]=gray
	# compute the absolute difference between the current frame and
	# first frame
	frameDelta1 = cv2.absdiff(preframe[0], preframe[2])
	thresh1 = cv2.threshold(frameDelta1, 25, 255, cv2.THRESH_BINARY)[1]
	frameDelta2 = cv2.absdiff(preframe[2], preframe[4])
	thresh2 = cv2.threshold(frameDelta2, 25, 255, cv2.THRESH_BINARY)[1]
	thresh1 = cv2.dilate(thresh1, None, iterations=6)
	thresh2 = cv2.dilate(thresh2, None, iterations=6)
	thresh=thresh1*thresh2
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	#thresh = cv2.erode(thresh, None, iterations=2)
	#thresh = cv2.dilate(thresh, None, iterations=6)
	(qwer,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
		#print cv2.contourArea(c)
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		#preframe=(x,y,w,h)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta1", frameDelta1)
	cv2.imshow("Frame Delta2", frameDelta2)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
	#firstFrame=gray
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
