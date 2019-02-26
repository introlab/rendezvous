import cv2
import sys
import face_detection

# Takes an image as argument, displays the image with rectangles around faces

framePath = sys.argv[1]
frame = cv2.imread(framePath)
faces = face_detection.detectFaces(frame)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", frame)
cv2.waitKey(0)