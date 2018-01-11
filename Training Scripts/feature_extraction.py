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

#Store descriptors
descriptors = []
#images = []

# Connect to the database.
conn = pymysql.connect(db='images_db', user='mldm_gangster', passwd='$aint3tienne', host='mldm-cv-project.cnpjv4qug6jj.us-east-2.rds.amazonaws.com')
#conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

sql_insert_desc = "INSERT INTO `desc_img` (`num_img`,`desc_json`) VALUES (%s, %s)"
#sql_insert_desc = "INSERT INTO `desc_obj` (`id_obj`,`desc_json`) VALUES (%s, %s)"

#Set directory path for image folders
folder="C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/JPEGImages/"

#Track run time
start=time.time()

cpt = 0

#Iterate through images extracting key poins and descriptors
for filename in os.listdir(folder):
    if filename != 'desktop.ini':
        #Import image
        img = cv2.imread(os.path.join(folder,filename))
    
        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #load feature detection algorithms
        #brisk=cv2.BRISK_create()
        sift = cv2.xfeatures2d.SIFT_create()
  
        #Run feature detection algorithm
        #k_brisk = brisk.detect(gray,None)
        k_sift = sift.detect(gray,None)
       
        #Computer keypoints and descriptors
        #k_brisk,d_brisk = brisk.compute(gray,k_brisk)
        k_sift,d_sift = sift.compute(gray,k_sift)
        #if d_brisk is not None:
        if d_sift is not None:
            #print(filename)
            #print(filename[0:filename.index('.',5)])
            #Store the keypoints and descriptors
            #json_desc  = json.dumps({'obj':filename[4:filename.index('_',5)],'brisk':d_brisk.tolist(),'sift':d_sift.tolist()})
            json_desc  = json.dumps({'img':filename,'sift':d_sift.tolist()})
                
        with conn.cursor() as cursor:
            cursor.execute(sql_insert_desc,(filename[0:filename.index('.',5)],json_desc)) #We execute our SQL request
            conn.commit()
        cpt += 1
        if cpt%100 == 0:
            print((cpt*100)/5011)
        
        
        

#End timer and show run time
end=time.time()
print(end - start)
