# -*- coding: utf-8 -*-
"""
Split images into folders according to their class

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

from shutil import copyfile
import pymysql
import pymysql.cursors
import os

# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

sql_get_labels = "SELECT `name` FROM `objects` WHERE `num_img`=%s"

cpt =0

for filename in os.listdir("C://Users//jerem//Desktop//M2//CV//VOCdevkit//VOC2007//JPEGImages//"):
    dictionnaire = {}
    dictionnaire["aeroplane"] = 0
    dictionnaire["bicycle"] = 0
    dictionnaire["bird"] = 0
    dictionnaire["boat"] = 0
    dictionnaire["bottle"] = 0
    dictionnaire["bus"] = 0
    dictionnaire["car"] = 0
    dictionnaire["cat"] = 0
    dictionnaire["chair"] = 0
    dictionnaire["cow"] = 0
    dictionnaire["diningtable"] = 0
    dictionnaire["dog"] = 0
    dictionnaire["horse"] = 0
    dictionnaire["motorbike"] = 0
    dictionnaire["person"] = 0
    dictionnaire["pottedplant"] = 0
    dictionnaire["sheep"] = 0
    dictionnaire["sofa"] = 0
    dictionnaire["train"] = 0
    dictionnaire["tvmonitor"] = 0
    with conn.cursor() as cursor:
        cursor.execute(sql_get_labels,(filename)) #We execute our SQL request
        conn.commit()
        for row in cursor:
            if dictionnaire[row[0]] == 0:
                if not os.path.exists("C:\\Users\\jerem\\Documents\\data\\"+row[0]):
                    os.makedirs("C:\\Users\\jerem\\Documents\\data\\"+row[0])
                copyfile("C:\\Users\\jerem\\Desktop\\M2\\CV\\VOCdevkit\\VOC2007\\JPEGImages\\"+filename, "C:\\Users\\jerem\\Documents\\data\\"+row[0]+"\\"+filename)
                dictionnaire[row[0]] = 1
#copyfile(src, dst)
