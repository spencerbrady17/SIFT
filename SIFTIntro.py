import numpy as np
import cv2 as cv
#https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html

# reading the image
img = cv.imread('AANHOX01.png')
# convert to greyscale
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# create SIFT feature extractor
sift = cv.SIFT_create()
# detect features from the image
kp = sift.detect(gray,None)

# draw the detected key points
img=cv.drawKeypoints(gray,kp,img)

# save the image
cv.imwrite('sift_keypoints.jpg',img)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg',img)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

