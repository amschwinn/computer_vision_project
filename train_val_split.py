# -*- coding: utf-8 -*-
"""
MLDM Computer Vision Project

Split images training set into 
test and validation sets

By: Austin Schwinn
Dec 9, 2017
"""
import os
import random
from math import floor
from PIL import Image

#%%
os.chdir('D:/GD/MLDM/Computer_Vision_project/cnn5/data')

for folder in os.listdir(os.path.join(os.getcwd(),'training')):
    images = os.listdir(os.path.join(os.getcwd(),'training',folder))
    for image in random.sample(images,floor(len(images)*.1)):
        img = Image.open(os.path.join(os.getcwd(),'training',folder,image))
        img.save(os.path.join(os.getcwd(),'test',folder,image),'PNG')
        os.remove(os.path.join(os.getcwd(),'training',folder,image))


