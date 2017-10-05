"""
Computer Vision Project
Feature extraction using 
BRISK and SIFT.
Object recognition
using Bag-Of-Words approach
and CNN

By: Austin Schwinn, 
Jeremie Blanchard,
Oussama Bouldjedri

October 3, 2017
"""

import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#Store key points and descriptions
kp_brisk = [] 
dsc_brisk = []
kp_sift = [] 
dsc_sift = []
#images = []

#Set directory path for image folders
folder="C:/Users/schwi/Google Drive/MLDM/Computer Vision Project/Data/trainVal_VOC2007/JPEGImages"    

#Track run time
start=time.time()

#Iterate through images extracting key poins and descriptors
for filename in os.listdir(folder):
    #Import image
    img = cv2.imread(os.path.join(folder,filename))
    
    #Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #load feature detection algorithms
    brisk=cv2.BRISK_create()
    sift = cv2.xfeatures2d.SIFT_create()

    #Run feature detection algorithm
    k_brisk = brisk.detect(gray,None)
    k_sift = sift.detect(gray,None)
    
    #Computer keypoints and descriptors
    k_brisk,d_brisk = brisk.compute(gray,k_brisk)
    k_sift,d_sift = brisk.compute(gray,k_sift)

    #Store the keypoints and descriptors
    kp_brisk.append(k_brisk)     
    dsc_brisk.append(d_brisk)
    kp_sift.append(k_sift)     
    dsc_sift.append(d_sift)

    #Draw and show kepoints on image
    #img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
    #plt.imshow(img2),plt.show()
   
#End timer and show run time
end=time.time()
print(end - start)

