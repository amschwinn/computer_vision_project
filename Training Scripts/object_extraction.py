# -*- coding: utf-8 -*-
"""
Extract objects from images

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

import pymysql
import pymysql.cursors
import os

import cv2
#import numpy as np


# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

#Create the standard MySQL request to obtain informations about objects in the given image
sql_get_info_object = "SELECT * FROM `objects` WHERE `num_img`=%s "

path = "C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/JPEGImages/"
image = "009822.jpg" #Name of the image we want to rextract objects from


for filename in os.listdir(path):
    
    #Import image
    img = cv2.imread(os.path.join(path,filename))
    
    image_name = filename[0:6]
    
    with conn.cursor() as cursor:
        cursor.execute(sql_get_info_object,filename) #We execute our SQL request
        conn.commit()
        for row in cursor: #Extract every object and save them as images named as follow : obj_"ID_Object"_"num-img".png
            obj = img[row[7]:row[9],row[6]:row[8]]
            cv2.imwrite('obj/obj_'+str(row[0])+'_'+image_name+'.png',obj)

