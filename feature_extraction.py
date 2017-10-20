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

import cv2
import os
import time
import pymysql
import pymysql.cursors
import json
import numpy as np
#Store descriptors
descriptors = []
#images = []

# Connect to the database.
#conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

#sql_insert_desc = "INSERT INTO `desc_obj` (`id_obj`,`desc_json`) VALUES (%s, %s)"

#Set directory path for image folders
#folder="C:/Users/jerem/Documents/obj/"
folder="/home/dell1/Desktop/f"

#Track run time
start=time.time()
#final=np.array()
#Iterate through images extracting key poins and descriptors
#for filename in os.listdir(folder):
#    if filename != 'desktop.ini':
        #Import image

MIN_MATCH_COUNT = 10

img1 = cv2.imread('4.jpg',0) # queryImage                                  ## we need to make a loop for all traning set
img2 = cv2.imread('5.jpg',0) # trainImage

# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)              ## change this to play with the metric 

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:                     ## change this to play with the metric
    good.append(m)


list_kp1 = []
list_kp2 = []
list2_kp1 =[]
list2_kp2 = []
        

for mat in good:
#Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    list2_kp1.append(img1_idx)
    img2_idx = mat.trainIdx
    list2_kp2.append(img2_idx)

print("matched indexes in key points found in image1")
print(list2_kp1)
print()
print("matched indexes in key points found in image2")
print(list2_kp2)

'''
##methode 2 less efficent(more time ,less tuning options )
img1 = cv2.imread('4.jpg',0)          # queryImage
img2 = cv2.imread('5.jpg',0) # trainImage

# Initiate SIFT detector

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1,des2)
list_kp1 = []
list_kp2 = []

# For each match...
for mat in matches:
    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    list_kp1.append(img1_idx)
    img2_idx = mat.trainIdx
    list_kp2.append(img2_idx)

print("indexes of matched key points found in image1")
print(list_kp1)
print()
print("indexes of matched key points found in image2")
print(list_kp2)

# Sort them in the order of their distance.
matches2 = sorted(matches, key = lambda x:x.distance)
list2_kp1 = []
list2_kp2 = []

# For each match...
for mat in matches2:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    list2_kp1.append(img1_idx)
    img2_idx = mat.trainIdx
    list2_kp2.append(img2_idx)

# For each match...


print()
print("indexes of keys points found in image1  ")
print(list2_kp1)
print()
print("indexes of matched key points found in image2 ")
print(list2_kp2)

'''

'''
        img = cv2.imread(os.path.join(folder,filename))
    
        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #load feature detection algorithms
        #brisk=cv2.BRISK_create()
        sift = cv2.xfeatures2d.SIFT_create()
        #sift = cv2.SIFT_create()
        #sift=cv2.BRISK_create()
  
        #Run feature detection algorithm
        #k_brisk = brisk.detect(gray,None)
        k_sift = sift.detect(gray,None)
       
        #Computer keypoints and descriptors
        #k_brisk,d_brisk = brisk.compute(gray,k_brisk)
        k_sift,d_sift = sift.compute(gray,k_sift)
        #if d_brisk is not None:
        
        if d_sift is not None:
            print(filename)
            #print(filename[4:filename.index('_',5)])
            #Store the keypoints and descriptors
            #json_desc  = json.dumps({'obj':filename[4:filename.index('_',5)],'brisk':d_brisk.tolist(),'sift':d_sift.tolist()})
            #json_desc  = json.dumps({'obj':filename[4:filename.index('_',5)],'sift':d_sift.tolist()})
              
        #with conn.cursor() as cursor:
        #    cursor.execute(sql_insert_desc,(filename[4:filename.index('_',5)],json_desc)) #We execute our SQL request
        #    conn.commit()
        
 '''       

#End timer and show run time
end=time.time()
print(end - start)
