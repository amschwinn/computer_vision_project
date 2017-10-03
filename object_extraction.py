# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:26:59 2017
Objects extraction from a given image
@author: jerem
"""

import pymysql
import pymysql.cursors

import cv2
#import numpy as np


# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

#Create the standard MySQL request to obtain informations about objects in the given image
sql_get_info_object = "SELECT * FROM `objects` WHERE `num_img`=%s "

path = "C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/JPEGImages/"
image = "009822.jpg" #Name of the image we want to rextract objects from

img = cv2.imread(path+image,3) #We open the image

with conn.cursor() as cursor:
    cursor.execute(sql_get_info_object,image) #We execute our SQL request
    conn.commit()
    for row in cursor: #Extract every object and save them as images named as follow : obj_"ID_Object"_"num-img".png
        obj = img[row[7]:row[9],row[6]:row[8]]
        cv2.imwrite('obj_'+str(row[0])+'_'+row[1]+'.png',obj)

