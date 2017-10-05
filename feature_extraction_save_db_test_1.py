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
import pandas as pd
import cv2
import os
import time
import pymysql
import pymysql.cursors

#Store descriptors
descriptors = []
#images = []

# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

sql_insert_desc = "INSERT INTO `desc_obj` (`id_obj`, `name_desc`,`width_matrix`, `height_matrix`) VALUES (%s, %s,%s, %s)"

sql_get_info_desc = "SELECT * FROM `desc_obj` WHERE `id_obj`=%s "

sql_set_info_cell = "INSERT INTO `values_matrix_desc` (`id_desc`, `column_matrix`,`line_matrix`,`value_cell`) VALUES (%s, %s,%s,%s)"

#Set directory path for image folders
folder="C:/Users/jerem/Documents/test/"    

#Track run time
start=time.time()

#Iterate through images extracting key poins and descriptors
for filename in os.listdir(folder):
    if filename != 'desktop.ini':
        #Import image
        img = cv2.imread(os.path.join(folder,filename))
    
        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #load feature detection algorithms
        brisk=cv2.BRISK_create()
        sift = cv2.xfeatures2d.SIFT_create()
  
        #Run feature detection algorithm
        k_brisk = brisk.detect(gray,None)
        k_sift = sift.detect(gray,None)
       
        #Computer keypoints and descriptors
        k_brisk,d_brisk = brisk.compute(gray,k_brisk)
        k_sift,d_sift = brisk.compute(gray,k_sift)
        if type(d_brisk) is not None:
            print(filename)
            print(filename[4:filename.index('_',5)])
            
        #Store the keypoints and descriptors
        with conn.cursor() as cursor:
            cursor.execute(sql_insert_desc,(filename[4:filename.index('_',5)],'brisk',len(d_brisk),len(d_brisk[0]))) #We execute our SQL request
            cursor.execute(sql_insert_desc,(filename[4:filename.index('_',5)],'sift',len(d_sift),len(d_sift[0]))) #We execute our SQL request
            conn.commit()
            
            cursor.execute(sql_get_info_desc,filename[4:filename.index('_',5)]) #We execute our SQL request
            conn.commit()
            for row in cursor:
                if row[2]== "sift":
                    for l in range(0,row[4]):
                        for c in range(0,row[3]):
                            cursor.execute(sql_set_info_cell,(row[0],c,l,int(d_sift[c][l]))) #We execute our SQL request
                            conn.commit()
                #elif row[2]== "brisk" :
                    

#End timer and show run time
end=time.time()
print(end - start)
