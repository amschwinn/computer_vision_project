# -*- coding: utf-8 -*-
"""
Load labels for test images

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

import xml.etree.ElementTree as ET
import os
import pandas as pd
import numpy as np

os.listdir('C:/Users/jerem/Desktop/M2/CV/VOCdevkit/VOC2007/Annotations')

#For all the XMLfile describing the images we are extracting the revelant informations and add them to the MySQL Database

df = pd.DataFrame()
test=[]
list_final_ys = {}
cpt=0
for xml_file in os.listdir('C:/Users/jerem/Desktop/M2/ML/VOCtestnoimgs_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'):

    tree = ET.parse("C:/Users/jerem/Desktop/M2/ML/VOCtestnoimgs_06-Nov-2007/VOCdevkit/VOC2007/Annotations/%s" % xml_file)
    root = tree.getroot()  
    for child in root:
        if child.tag == "filename":
            img_name = child.text
        elif child.tag == "object":
            obj_name = child.find("name").text
            test += [obj_name]
            #print("obj_name = ",obj_name)
    list_final_ys[img_name] = test       
    test=[]                
