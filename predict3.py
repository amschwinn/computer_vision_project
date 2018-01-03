#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:12:27 2017

@author: dell1
"""

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import cv2

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import os 
import xml.etree.ElementTree as ET
import os
import pandas as pd
import numpy as np 

def pause():
    programPause = input("Press the <ENTER> key to continue...")


target_size = (229, 229) #fixed size for InceptionV3 architecture
Classes=["areoplane","bycicle","bird","boat","bottle","bus","car","cat","chair",
         "cow","digningtable","dog","horse","motorbike","person","pottedplant","sheep",
         "sofa","train","TVmonitor"]

Classes2=["A","B","C","D","E,","F","G","H","I","J","K","L","M","N","O","P","K","R","S","T"]

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("cat", "dog")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image folder")
  a.add_argument("--image_url", help="url to image")
  a.add_argument("--model")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    trueresult=[] 
    test=[]
    predictedresult=[]
    for subdirs, dirs, files in os.walk(args.image):  
  
     for file in files:  
        #if file == '*.shp':  
      print("file is ",file)
      fileonly=os.path.splitext(file)[0]
      xml_file=fileonly+".xml"
      #print("the xml file is ",xml_file)
      tree = ET.parse("/home/dell1/Desktop/cnn5/VOCdevkit/VOC2007/Annotations/%s" % xml_file)
      root = tree.getroot()  
      for child in root:
        if child.tag == "filename":
            img_name = child.text
        #    print("image name is",img_name )
            #temp=[]
        elif child.tag == "object":
            obj_name = child.find("name").text
            if obj_name  not in test:
             test += [obj_name]
            #test2+=[obj_name]
       #     print("obj_name = ",obj_name)
            print("test is ",test)
      
      teststr=""
      #teststr2=str(teststr)
      for i in range (1 ,20):  
       if Classes[i] in test:
           teststr+=Classes2[i]
      print("the hot ecnoding is ",teststr)     
      #list_final_ys[img_name] = test2 
      trueresult.append(teststr)      
      test=[]  
      #test2=[]
      #print("true result is ",trueresult)    
      
      
      
      
      
      
      
      #path=args.image+'/'+file
      path=subdirs+'/'+file
      #print("path is ",path)
      
      
      
      
      
      
      
      
      #img = Image.open(args.image)
      img = Image.open(path)
      #print("foo")
      preds = predict(model, img, target_size)
      #print(preds)
      print()
      objs=Classes
    #print("objects are",objs)
    #Percent of total prediction by each object type
      percent_pred = []
      for i in preds:
       percent_pred = percent_pred + [(i/np.sum(preds))]
      #print(percent_pred) 
      print()
      #print()
    #combine percents with labels
    #percent_pred = pd.DataFrame(percent_pred[0], index=objs)
      result=np.column_stack((percent_pred,objs))
      print(result)
      print()
      preds[preds>=0.2] = 1
      preds[preds<0.2] = 0
      res=[]
      for i in range (1,20):
        if (preds[i]==1):
         res.append(objs[i])
      print(res)
      teststrpred=""
      #teststrpred2=str(teststrpred)
      for i in range (1 ,20):  
       if Classes[i] in res:
           teststrpred+=Classes2[i]
      print("the hot ecnoding is ",teststrpred)   
      
      
      
      #predictedresult.append(res)
      predictedresult.append(teststrpred)
      #for i in range(1,20):
      #    if objs[i] in res :
      #        res2=
      #pause()
      
      #plot_preds(img, preds)
      #ind=np.argmax(preds)
      
      #print("there is a probability of ",preds[ind]*100)
      #print(" this object is from the class",Classes[ind])
      #print()
    '''
    orig = cv2.imread(args.image)
   #(imagenetID, label, prob) =
    label=Classes[ind]
    prob=preds[ind]*100
    #cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob),
	#(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #cv2.imshow("Classification", orig)
    #cv2.waitKey(0)
    '''
  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

