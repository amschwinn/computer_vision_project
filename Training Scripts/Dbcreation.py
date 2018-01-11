
# -*- coding: utf-8 -*-
"""
Create a database of the extracted object images

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

import pymysql
import pymysql.cursors
import xml.etree.ElementTree as ET
import os
 
# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

os.listdir('C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/Annotations')

#Definition of the SQL request to add objects and images
sql_add_img = "INSERT INTO `images` (`num_img`, `flickr_id`,`width`, `height`,`depth`) VALUES (%s, %s, %s, %s, %s)"
sql_add_object = "INSERT INTO `objects` (`num_img`, `name`,`pose`, `truncated`,`difficulty`,`xmin`,`ymin`,`xmax`,`ymax`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

#For all the XMLfile describing the images we are extracting the revelant informations and add them to the MySQL Database
for xml_file in os.listdir('C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/Annotations'):

    tree = ET.parse("C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/Annotations/%s" % xml_file)
    root = tree.getroot()
    
    with conn.cursor() as cursor:
        for child in root:
            if child.tag == "filename":
                img_name = child.text
            elif child.tag == "source":
                flickrid = child.find("flickrid").text
            elif child.tag == "size":
                width_img = child.find("width").text
                height_img = child.find("height").text
                depth_img = child.find("depth").text
                cursor.execute(sql_add_img, (img_name,flickrid,width_img,height_img,depth_img))
            elif child.tag == "object":
                obj_name = child.find("name").text
                obj_pose = child.find("pose").text
                obj_truncated = child.find("truncated").text
                obj_difficulty = child.find("difficult").text
                obj_xmin = child.find("bndbox").find("xmin").text
                obj_ymin = child.find("bndbox").find("ymin").text
                obj_xmax = child.find("bndbox").find("xmax").text
                obj_ymax = child.find("bndbox").find("ymax").text
                cursor.execute(sql_add_object, (img_name,obj_name,obj_pose,obj_truncated,obj_difficulty,obj_xmin,obj_ymin,obj_xmax,obj_ymax))
        conn.commit()                             
        
conn.close()