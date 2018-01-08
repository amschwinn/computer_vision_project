#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:11:13 2017

@author: dell1
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from PIL import Image

from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
#%matplotlib inline
print("##############################################################")
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input



import numpy as np
with tf.device('/gpu:1'):
    model = VGG16(weights='imagenet', include_top=False)
    nb_features = 51200
    f = np.empty(nb_features) ## change to the nbr of all images 
    labels = []
    #labels=np.empty([20])
    #labels= np.array([])
    img_path ='C:/Users/jerem/Documents/data/'
    #img_path = 'D:/GD/MLDM/Computer_Vision_Project/cnn5/data/Objects/training'
    #img_path = 'D:/GD/MLDM/Computer_Vision_Project/cnn5/data/training'
    
    for subdirs, dirs, files in os.walk(img_path):
         print("pass")
         path=[]
         print("the current sub dir is ",subdirs)
         print("the current dir is ",dirs)
         for file in files:  
          path=subdirs+'/'+file
          img = image.load_img(path, target_size=(224, 224))
          x = image.img_to_array(img)
          x = np.expand_dims(x, axis=0)
          x = preprocess_input(x)
    
          features3 = model.predict(x)

          flat3=features3.flatten()

          f = np.append(f,flat3)
          currentdir=os.path.basename(os.path.normpath(subdirs))

          labels.append(currentdir)
#%%    
f.shape = (int(len(f)/nb_features),nb_features)
f = f[1:,:]
X=f
Y=labels   

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=42)

###############################################################SVM part###
corresp = {}
corresp[0] = "aeroplane"
corresp[1] = "bicycle"
corresp[2] = "bird"
corresp[3] = "boat"
corresp[4] = "bottle"
corresp[5] = "bus"
corresp[6] = "car"
corresp[7] = "cat"
corresp[8] = "chair"
corresp[9] = "cow"
corresp[10] = "diningtable"
corresp[11] = "dog"
corresp[12] = "horse"
corresp[13] = "motorbike"
corresp[14] = "person"
corresp[15] = "pottedplant"
corresp[16] = "sheep"
corresp[17] = "sofa"
corresp[18] = "train"
corresp[19] = "tvmonitor"

from sklearn.externals import joblib
from imblearn.metrics import sensitivity_score,geometric_mean_score
for k in range(0,20):
    y_train_tmp = np.zeros(len(y_train))
    for i in range(0,len(y_train)):
        if y_train[i] == corresp[k]:
            y_train_tmp[i] = 1
        else:
            y_train_tmp[i] =0

    svm_obj = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')    
    svm_obj.fit(X_train, y_train_tmp)
    
    name_file_svm = "svm_vgg_"+corresp[k]+".pkl"

    joblib.dump(svm_obj, name_file_svm) 
    
    y_test_tmp = np.zeros(len(y_test))
    
    for j in range(0,len(y_test)):
        if y_test[j] == corresp[k]:
            y_test_tmp[j] = 1
        else:
            y_test_tmp[j] =0
    y_preds = svm_obj.predict(X_test)
    
    acc=accuracy_score(y_test_tmp, y_preds)
    acc_avg= average_precision_score(y_test_tmp, y_preds)
    print("Sensitivity Score : ",sensitivity_score(y_test_tmp, y_preds, average='macro'))
    print("Geometric mean Score : ",geometric_mean_score(y_test_tmp, y_preds))
    print("Accuracy average SVM_",corresp[k]," is ",acc_avg)
    print("Accuracy SVM_",corresp[k]," is ",acc)
    print(confusion_matrix(y_test_tmp, y_preds))


 
#%%

#from sklearn.metrics import average_precision_score
#b2=average_precision_score(y_test, y_preds) 
#print("the average precision for noemal SVM is ",b2)

#%%

clf2=OneVsRestClassifier(LinearSVC(C=100.)).fit(X_train, y_train)
y_preds2=clf2.predict(X_test)
b3=accuracy_score(y_test, y_preds2)
print("the accuracy score for one class SVM is   ",b3) 
#b4=average_precision_score(y_test, y_preds2) 
#print("the average precision for  svm once class is ",b4)
#%%

from sklearn.model_selection import cross_val_score
from sklearn import model_selection
seed = 7

kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
clf3 = OneVsRestClassifier(LinearSVC(C=100.)).fit(X,Y)
scores = cross_val_score(clf3, X, Y,cv=kfold)
b5=scores.mean()
print("the Kfold cross validation score for thr one class SVM is ",b5)
#b6=average_precision_score(y_test, y_preds2) 
#print(b6)

#%%
############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
seed = 7

kfold3 = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
clf5 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
clf5.fit(X_train, y_train)
y_preds4=clf5.predict(X_test)
b9=accuracy_score(y_test, y_preds4)
print("the accuracy score for desicion tree  is ",b9) 

scores3 = cross_val_score(clf5, X, Y,cv=kfold3)
b10=scores3.mean()
print("the kfold score for desicion tree is ",b10)
#b11=average_precision_score(y_test, y_preds4) 
#print("the average precision score for desicion tree  is ", b11)

########################################################################
#%%
#import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

seed = 7
num_trees = 100
max_features = 200
kfold4 = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train, y_train)
y_preds5=model.predict(X_test)
b12=accuracy_score(y_test, y_preds5)
print("the accuracy score for extra tree is ",b12) 


results = model_selection.cross_val_score(model, X, Y,cv=kfold4)
b13=results.mean()
print("the k fold cross val score for extra tress is ", b13)

#b14=average_precision_score(y_test, y_preds5) 
#print("the average precision score for extra tress is ",b14)

#%%
from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 100
kfold5 = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
model2 = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model2.fit(X_train, y_train)
y_preds6=model2.predict(X_test)
b12=accuracy_score(y_test, y_preds6)
print("the accuracy score for Ada Boost is ",b12) 
#b13=average_precision_score(y_test, y_pred) 
#print(b13)
results = model_selection.cross_val_score(model2, X, Y,cv=kfold5)
b14=results.mean()
print("the kfold cross validation score for ada boost is ",b14)
#b13=average_precision_score(y_test, y_preds6) 
#print("the verage presicion score for ada boost is ",b13)
######################################
#%%
from skmultilearn.adapt import MLkNN
clf4=MLkNN(k=1)
#
kfold2 = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
scores2 = cross_val_score(clf4, X, Y,cv=kfold2)
b8=scores2.mean()
print("kfold for MLKNN is ",b8)
clf4.fit(X_train,y_train)
y_preds3=clf4.predict(X_test)
b7=accuracy_score(y_test,y_preds3)
print("accuracy in MLKNN is ",b7)







#%%
'''
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(neigh, X, Y,cv=kfold)
print(results.mean())

#%%
import os
 
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
 
#def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
 
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
 
param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]
 
    # request probability estimation
svm = SVC(probability=True)
 
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
clf = grid_search.GridSearchCV(svm, param,cv=10, n_jobs=4, verbose=3)
clf.fit(X_train, y_train)
if 
  os.path.exists(model_output_path):joblib.dump(clf.best_estimator_, model_output_path)
else:
  print("Cannot save trained svm model to {0}.".format(model_output_path))
 
print("\nBest parameters set:")
print(clf.best_params_)
y_predict=clf.predict(X_test)
labels=sorted(list(set(labels)))
print("\nConfusion matrix:")
print("Labels: {0}\n".format(",".join(labels)))
print(confusion_matrix(y_test, y_predict, labels=labels))
#print("\nClassification report:")
#print(classification_report(y_test, y_predict)) 
'''