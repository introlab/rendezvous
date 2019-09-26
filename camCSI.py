import cv2
import numpy as np

#stream = cv2.VideoCapture(0)  
stream = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=UYVY,width=640,height=480 ! videoconvert ! video/x-raw, format=BGR ! appsink")  
#stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 

while True:
	ret, frame = stream.read();
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	font = cv2.imshow('CSI Video', frame)
	
	if cv2.waitKey(1)  & 0xFF == ord('q'):
		break

stream.release()
cv2.destroyAllWindows()
