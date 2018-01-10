#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to find CNN accuracy, precision, and recall
"""

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os 
import xml.etree.ElementTree as ET
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

#%%
'''
Classes=["areoplane","bycicle","bird","boat","bottle","bus","car","cat","chair",
         "cow","digningtable","dog","horse","motorbike","person","pottedplant","sheep",
         "sofa","train","TVmonitor"]
'''
#Set arguments
target_size = (229, 229) #fixed size for InceptionV3 architecture
threshold = .5
Classes = os.listdir('D:/GD/MLDM/Computer_Vision_Project/cnn5/data/training')
model = 'D:/GD/MLDM/Computer_Vision_Project/github/cvp_cnn_sigmoid_10_10.model'
model = load_model(model)
test_dir = 'D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/JPEGImages/'
test_ann_dir = "D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/Annotations2/"

#%%
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

#%%
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

#%%
#Get all test images
test_imgs = os.listdir(test_dir)
test_anns = os.listdir(test_ann_dir)
pred_class = pd.DataFrame(index=Classes)
true_class = pd.DataFrame(index=Classes)

#Iterate and get prediction and correct class tables
for img_name in test_imgs:
    img_name = img_name[:-4]
    #Ensure we have correct label values
    if (img_name+'.xml') in test_anns:
        
        #Load image
        img = Image.open(test_dir+img_name+'.jpg')
    
        #Predict labels
        preds = predict(model, img, target_size)
    
        #Percent of total prediction by each object type  
        percent_pred = []
        for i in preds:
            percent_pred = percent_pred + [(i/np.sum(preds))]  
        
        #combine percents with labels
        percent_pred = pd.DataFrame(percent_pred, index=Classes, columns=[img_name])
    
        pred_class = pred_class.join(percent_pred)
        
        print(img_name)
        
        #Get correct labels
        tree = ET.parse(test_ann_dir + img_name + '.xml')
        root = tree.getroot() 
        
        class_names = []                
        for child in root:
            if child.tag == "object":
                obj_name = child.find("name").text
                if obj_name  not in class_names:
                    class_names += [obj_name]
        
        #Create one hot encoding
        one_hot = pd.DataFrame(np.repeat(0,20),index=Classes,columns=[img_name])
        for class_name in class_names:
            one_hot.loc[class_name,img_name] = 1
        true_class = true_class.join(one_hot)
        

#%%%
#pred_class_backup = pred_class.copy() 
pred_class = pred_class_backup.copy()
#%%
pred_class.iloc[0:20,0:20]
true_class.iloc[0:20,0:20]
#%%
#Perfect of total prediction by class
for col in pred_class.columns:
    col_sum = pred_class.loc[:,col].sum()
    for row in pred_class.index:
        pred_val = pred_class.loc[row,col].astype('float64')
        pred_class.loc[row,col] = pred_val/col_sum
        '''
        if pred_val == 0:
            pred_class.loc[row,col] = 0
        else:
            pred_class.loc[row,col] = pred_val/col_sum
        '''
#%%
results = pd.DataFrame(columns=['tp','fn','tn','fn','acc','prec',
                                'rec'])        
        
for i in range(1,15,1):
    #pred_class_backup2 = pred_class.copy() 
    pred_class = pred_class_backup2.copy()
        
    #Use threshold to create binary classifications
    threshold = i/100
    print(threshold)
    for col in pred_class.columns:
        for row in pred_class.index:
            if pred_class.loc[row,col] >= threshold:
            #if pred_class.loc[row,col] != 0:
                pred_class.loc[row,col] = 1
            else:
                pred_class.loc[row,col] = 0
    pred_class = pred_class.astype('int')
            
    
    #Compare predictions vs true labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for y in pred_class.index:
        for x in pred_class.columns:
            true_val = true_class.loc[y,x] 
            pred_val = pred_class.loc[y,x]
            if ((true_val==1) & (pred_val==1)):
                tp += 1
            elif ((true_val==0) & (pred_val==1)):
                fp += 1
            elif ((true_val==1) & (pred_val==0)):
                fn += 1
            elif ((true_val==0) & (pred_val==0)):
                tn += 1
    #Store results
    results.loc[threshold,'tp'] = tp
    results.loc[threshold,'fp'] = fp
    results.loc[threshold,'tn'] = tn
    results.loc[threshold,'fn'] = fn
    results.loc[threshold,'acc'] = ((tp+tn)/(tp+fp+tn+fn))
    results.loc[threshold,'prec'] = (tp/(tp+fp))
    results.loc[threshold,'rec'] = (tp/(tp+fn))
    print(tp)
    print(fp)
    print(tn)
    print(fn)
    #Accuracy, precision, recall
    print((tp+tn)/(tp+fp+tn+fn))
    print(tp/(tp+fp))
    print(tp/(tp+fn))
    print('#######################################')
    
  
results.to_csv('NN_results.csv')
        
        
        


#%%
'''
if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  a.add_argument("--model")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    print(preds)
    print()
    #objs = os.listdir('/home/dell1/Desktop/cnn5/data/training')
    objs=Classes
    #print("objects are",objs)
    #Percent of total prediction by each object type
    percent_pred = []
    for i in preds:
      percent_pred = percent_pred + [(i/np.sum(preds))]
    print(percent_pred) 
    print()
    print()
    #combine percents with labels
    #percent_pred = pd.DataFrame(percent_pred[0], index=objs)
    result=np.column_stack((percent_pred,objs))
    print(result)
    print()
    print(result[np.lexsort(np.fliplr(result).T)])
    #plot_preds(img, preds)
    #ind=np.argmax(preds)
    #print("there is a probability of ",preds[ind]*100)
    #print(" this object is from the class",Classes[ind])
    
    
    #orig = cv2.imread(args.image)
   #(imagenetID, label, prob) =
    #label=Classes[ind]
    #prob=preds[ind]*100
    #cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob),
	#(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #cv2.imshow("Classification", orig)
    #cv2.waitKey(0)

  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)
'''
