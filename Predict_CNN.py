#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using CNN to create descriptors and neural layer to predict
object recognition in images.

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""
import os
###############################################################################
#Set Params
Classes = os.listdir('D:/GD/MLDM/Computer_Vision_Project/cnn5/data/training')
model = 'D:/GD/MLDM/Computer_Vision_Project/cnn5/results/cvp_cnn_sigmoid_10_10.model'
test_dir = 'D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/JPEGImages/'
test_ann_dir = "D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/Annotations2/"
results_file = 'D:/GD/MLDM/Computer_Vision_Project/cnn5/results/cvp_cnn_sigmoid_10_10_results.csv'
specific_img = None #[]
###############################################################################



#%%
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
 


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
  #return x
  
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
#Percent of total prediction by class
def pred_percent(pred_class):
    for col in pred_class.columns:
        col_sum = pred_class.loc[:,col].sum()
        for row in pred_class.index:
            pred_val = pred_class.loc[row,col].astype('float64')
            pred_class.loc[row,col] = pred_val/col_sum
    return pred_class

#%%
#Use threshold to create binary prediction matrix
def binary_pred(pred_class, threshold):
    #Use threshold to create binary classifications
    for col in pred_class.columns:
        for row in pred_class.index:
            if pred_class.loc[row,col] >= threshold:
            #if pred_class.loc[row,col] != 0:
                pred_class.loc[row,col] = 1
            else:
                pred_class.loc[row,col] = 0
    pred_class = pred_class.astype('int')
    
    return pred_class
#%%
def make_prediction(test_dir,test_ann_dir,target_size,model,specific_img=None):
    #Get all test images
    test_imgs = os.listdir(test_dir)
    test_anns = os.listdir(test_ann_dir)
    pred_class = pd.DataFrame(index=Classes)
    true_class = pd.DataFrame(index=Classes)
    
    if specific_img:
        test_imgs = [x for x in test_imgs if x in specific_img]
    
    #Iterate and get prediction and correct class tables
    print('Predicting')
    preds = []
    for img_name in test_imgs:
        img_name = img_name[:-4]

        #Ensure we have correct label values
        if (img_name+'.xml') in test_anns:            
            #Load image
            img = Image.open(test_dir+img_name+'.jpg')

            #Predict labels
            preds += [predict(model, img, target_size)]
        print(img_name)   
    #return preds
    
    print('Testing')        
    for j in range(len(preds)):
            pred = preds[j]
            img_name = test_imgs[j]
            img_name = img_name[:-4]

            print(img_name)
            #Percent of total prediction by each object type  
            percent_pred = []
            for i in pred:
                percent_pred = percent_pred + [(i/np.sum(pred))]  
            
            #combine percents with labels
            percent_pred = pd.DataFrame(percent_pred, index=Classes, columns=[img_name])
        
            pred_class = pred_class.join(percent_pred)
    
    #Get percent of prediction for each class
    pred_class = pred_percent(pred_class)
    
    #Use threshold to get binary classification
    pred_class = binary_pred(pred_class, threshold=.26)
    
    print('Compiling correct labels')
    for img_name in test_imgs:
        img_name = img_name[:-4]
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
        '''
        #Print prediction vs actual
        for x in true_class.columns:
            print('#######################################')
            print('Image: ' + str(x))
            print('***************************************')
            print('Predicted Labels:')
            for y in true_class.index:
                print(str(y) + ': ' + str(pred_class.loc[y,x]))                
            print('***************************************')
            print('True Labels:')
            for y in true_class.index:
                print(str(y) + ': ' + str(true_class.loc[y,x]))                
            print('***************************************')                
        '''    
    results = pd.DataFrame(columns=['tp','fp','tn','fn','acc','prec','rec'])           
    #Compare predictions vs true labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for y in pred_class.index:
        temp_tp = 0
        temp_fp = 0
        temp_tn = 0
        temp_fn = 0
        for x in pred_class.columns:
            true_val = true_class.loc[y,x] 
            pred_val = pred_class.loc[y,x]
            if ((true_val==1) & (pred_val==1)):
                tp += 1
                temp_tp += 1
            elif ((true_val==0) & (pred_val==1)):
                fp += 1
                temp_fp += 1
            elif ((true_val==1) & (pred_val==0)):
                fn += 1
                temp_fn += 1
            elif ((true_val==0) & (pred_val==0)):
                tn += 1
                temp_tn += 1
        results.loc[y,'tp'] = temp_tp
        results.loc[y,'fp'] = temp_fp
        results.loc[y,'tn'] = temp_tn
        results.loc[y,'fn'] = temp_fn
        results.loc[y,'acc'] = ((temp_tp+temp_tn)/(temp_tp+temp_fp+temp_tn+temp_fn))
        if (temp_tp+temp_fp) > 0:
            results.loc[y,'prec'] = (temp_tp/(temp_tp+temp_fp))
        if (temp_tp+temp_fn) > 0:
            results.loc[y,'rec'] = (temp_tp/(temp_tp+temp_fn))    
    #Results
    print('True Positives: ' + str(tp))
    print('False Positives: ' + str(fp))
    print('True Negatives: ' + str(tn))
    print('False Negatives: ' + str(fn))
    #Accuracy, precision, recall
    print('Accuracy: ' + str((tp+tn)/(tp+fp+tn+fn)))
    print('Precision: ' + str(tp/(tp+fp)))
    print('Recall: ' + str(tp/(tp+fn)))
    print('#######################################')
    #results = pd.DataFrame(columns=['tp','fp','tn','fn','acc','prec','rec']) 
    results.loc['Total','tp'] = tp
    results.loc['Total','fp'] = fp
    results.loc['Total','tn'] = tn
    results.loc['Total','fn'] = fn
    results.loc['Total','acc'] = ((tp+tn)/(tp+fp+tn+fn))
    results.loc['Total','prec'] = (tp/(tp+fp))
    results.loc['Total','rec'] = (tp/(tp+fn))
    results.to_csv(results_file)
            
    return pred_class, true_class
    
   
        
#%%
#Run Prediction
target_size = (299, 299) #fixed size for InceptionV3 architecture
print('loading model')
model = load_model(model)
print('model loading')
print('Starting prediction model')
pred_class, true_class = make_prediction(test_dir,test_ann_dir,target_size,model,specific_img)
