import numpy as np
import os
import cv2 as cv

imagepath = "..\Autopack\example_usage\motif_contour_plots"
imagelist = os.listdir(imagepath)

imagelist = [i for i in imagelist if i.endswith('npy')]

for i in imagelist:
    fullimagepath = os.path.join(imagepath, i)
    imagedata = np.load(fullimagepath)
    #print(imagedata.shape)        
    
    imagedata = cv.normalize(imagedata, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.SIFT_create(contrastThreshold=.0000001)
    kp = sift.detect(imagedata,None)

    img=cv.drawKeypoints(imagedata,kp,imagedata)
    cv.imwrite('sift_keypoints'+ i.strip('.npy') +'.jpg',img)

    print (kp)
    
   

# alpha = 1.95 # Contrast control (1.0-3.0)
# # beta = 0 # Brightness control (0-100)

# # manual_result = cv.convertScaleAbs(imagedata, alpha=alpha, beta=beta)

# imagedata = cv.normalize(imagedata, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
# #threshold_imagedata = cv.threshold(imagedata,0,255,cv.THRESH_TOZERO)  

  
# sift = cv.SIFT_create(contrastThreshold=.0000001)
# kp = sift.detect(imagedata,None)

# img=cv.drawKeypoints(imagedata,kp,imagedata)
# cv.imwrite('sift_keypoints.jpg',img)

# print (kp)