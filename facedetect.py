import cv2
import numpy as np

#Initialize video capture. if an integer is passed instead of a string
#the video source with that index is used
video = cv2.VideoCapture(0)
flag = True
#Create window to visualize results
cv2.namedWindow('faces')
#Initialize face detector object
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Main loop
while flag:
    #Load next frame
    flag, frame = video.read()
    #Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect faces as rectangles
    rects = face_detect.detectMultiScale(gray, scaleFactor=1.3,
                                     minNeighbors=4, minSize=(30, 30),
                                     flags = cv2.CASCADE_SCALE_IMAGE)
    #Copy frame for visualization
    img = frame.copy()
    green = (0, 255, 0)
    #For each detected face...
    for x1, y1, x2, y2 in rects:
        #Draw rectangle in duplicated image
        cv2.rectangle(img, (x1, y1), (x2+x1, y2+y1), green, 2)
    #Show in window
    cv2.imshow('faces', img)
    #Wait for keypress
    key = cv2.waitKey(5)
    if key == 'q':
        flag = False
sys.exit()
