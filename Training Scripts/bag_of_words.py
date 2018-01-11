# -*- coding: utf-8 -*-
"""
Using kmeans to create a bag of visual words and SVM to predict
object recognition in images.

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

#Set working directory
import os
os.chdir('D:/GD/MLDM/Computer_Vision_Project/github2')
#os.path.dirname(os.path.abspath(__file__))
os.getcwd()

import features_kmeans as km
import pandas as pd
import timeit
import numpy as np
import pickle
import cv2
import xml.etree.ElementTree as ET


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#%%
Classes = os.listdir('D:/GD/MLDM/Computer_Vision_Project/cnn5/data/training')
test_dir = 'D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/JPEGImages/'
test_ann_dir = "D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/Annotations2/"
train_dir = 'D:/GD/MLDM/Computer_Vision_Project/Data/trainVal_VOC2007/JPEGImages/'
train_ann_dir = "D:/GD/MLDM/Computer_Vision_Project/Data/trainVal_VOC2007/Annotations/"


#%%
print('Start Load')
start = timeit.default_timer()

#Load object descriptors
obj_desc_list, obj_id, obj_name = km.load_objects_desc(20000)

end = timeit.default_timer()
print('Load Complete')
print(end-start)

#%%
obj_total = [obj_desc_list, obj_id, obj_name]
pickle.dump(obj_total,open('D:/GD/MLDM/Computer_Vision_Project/obj_total.sav','wb'))    
#%%
obj_total = pickle.load(open('D:/GD/MLDM/Computer_Vision_Project/obj_total.sav','rb'))   
obj_desc_list = obj_total[0] 
obj_id = obj_total[1]  
obj_name   = obj_total[2] 

#%%
###############################################################################
# PCA
###############################################################################
print('Start PCA')
start = timeit.default_timer()

#Transform into df for pca
pca_obj_desc = pd.DataFrame(obj_desc_list)

#Find optimal principle components
#pca_results = km.pca_test(pca_obj_desc)

#Run PCA and transform to optimal number of PCs
n_comp = 50
pca = PCA(n_components=n_comp)
pca.fit(pca_obj_desc)
pca_obj_desc = pd.DataFrame(pca.transform(pca_obj_desc))

end = timeit.default_timer()
print('PCA Complete')
print(end-start)

pickle.dump(pca,open('D:/GD/MLDM/Computer_Vision_Project/obj_pca.sav','wb'))

#%%
pca = pickle.load(open('D:/GD/MLDM/Computer_Vision_Project/obj_pca.sav','rb'))
#Transform into df for pca
pca_obj_desc = pd.DataFrame(obj_desc_list)
pca_obj_desc = pd.DataFrame(pca.transform(pca_obj_desc))
  
#%%
###############################################################################
# K-Means
###############################################################################
print('Start K-Means')
start = timeit.default_timer()
'''
#Combine PCA results with obj name
pca_obj_desc['obj_name'] = pd.Series(obj_name)

#Run test kmeans results to find optimal k
#test_kmeans_results, test_centroids = km.kmeans_test(pca_obj_desc, 10, 100, 10)

#Run kmeans on object descriptors to get our visual bag of words for each obj
#Combine PCA results with obj name
pca_obj_desc['obj_name'] = pd.Series(obj_name)

obj_kmeans_results, obj_centroids = km.kmeans_per_object(pca_obj_desc,50)
'''

#Load bag of words from results filese
os.chdir('D:/GD/MLDM/Computer_Vision_Project/results')
obj_centroids = km.load_clusters('D:/GD/MLDM/Computer_Vision_Project/results')

#Combine all separate obj centroids into a single group of centroids for our 
#total bag of visual words
tot_obj_centroids = km.combine_clusters(obj_centroids)
tot_obj_centroids = tot_obj_centroids.iloc[:,50:]

#Create k-means model with the all object centroids 
#obj_kmeans_total = KMeans(init=tot_obj_centroids,
#                          n_clusters=len(tot_obj_centroids),
#                          n_init=1).fit(tot_obj_centroids)
obj_kmeans_total = KMeans(n_clusters=len(tot_obj_centroids)).fit(pca_obj_desc.iloc[:,0:50])

#Predict to get bag of words for svm training objects
obj_clusters_total = obj_kmeans_total.predict(pca_obj_desc.iloc[:,0:50])

#Format for svm training feed
obj_cluster_pivot = km.cluster_format(obj_clusters_total, obj_id,obj_name)

end = timeit.default_timer()
print('K-Means Complete')
print(end-start)

#%%
pickle.dump(obj_kmeans_total,open('D:/GD/MLDM/Computer_Vision_Project/obj_kmeans.sav','wb'))
#%%
obj_kmeans_total = pickle.load(open('D:/GD/MLDM/Computer_Vision_Project/obj_kmeans.sav','rb'))

#%%
###############################################################################
# SVM
###############################################################################
print('Start SVM')
start = timeit.default_timer()
'''
#split features and labels
X = obj_cluster_pivot.drop(['obj_name'],axis=1)
y = obj_cluster_pivot['obj_name']

#Normalize X
X = normalize(X,axis=1)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y)


#Use grid search to find correct hyperparameter
#Specify params to iterate over
tuned_params = [{'kernel': ['rbf','sigmoid'], 'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                {'kernel': ['poly'], 'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100, 1000], 'degree':[1,2,3,4,5,6,7]}]

scores = ['precision', 'recall','accuracy']

#Run grid search
svm_clf = GridSearchCV(svm.SVC(), tuned_params, cv=5)
svm_clf.fit(X, y)
svm_grid_results = pd.DataFrame(svm_clf.cv_results_)
svm_grid_results.to_csv('svm_gridsearch_results.csv')


end = timeit.default_timer()
print('SVM Complete')
print(end-start)


#Create and train SVM model
svm_clf = svm.SVC(kernel='linear').fit(X_train, y_train) 
y_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred)
'''
#%%
#Get frequency of each descripter
def desc_freq(clusters):
    freq = np.array(np.repeat(0,1000),dtype='float64').reshape((1,1000))
    code, counts = np.unique(clusters, return_counts=True)
    
    for j in range(len(code)):
        freq[0,code[j]] = counts[j]
    
    return freq
#%%
#Get all test images
test_imgs = os.listdir(train_dir)
test_anns = os.listdir(train_ann_dir)
pred_class = pd.DataFrame(index=Classes)
true_class = pd.DataFrame(index=Classes)

'''
if specific_img:
    test_imgs = [x for x in test_imgs if x in specific_img]
'''

#Iterate and get prediction and correct class tables
desc = np.array([np.repeat(0,1000)])
img_num = []
img_labels = []
for img_name in test_imgs:
    img_name = img_name[:-4]
    img_num = img_num + [img_name]
    #Ensure we have correct label values
    if (img_name+'.xml') in test_anns:
        
        print(img_name)
        #Load image
        #from PIL import Image
        #img = Image.open(t_dir+img_name+'.jpg')
        img = cv2.imread((train_dir+img_name+'.jpg'))
    
        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #load feature detection algorithms
        #brisk=cv2.BRISK_create()
        sift = cv2.xfeatures2d.SIFT_create()
  
        #Run feature detection algorithm
        #k_brisk = brisk.detect(gray,None)
        k_sift = sift.detect(gray,None)
       
        #Computer keypoints and descriptors
        #k_brisk,d_brisk = brisk.compute(gray,k_brisk)
        k_sift,d_sift = sift.compute(gray,k_sift)

        #Use pca to reduce dimensionality
        pca_desc = pd.DataFrame(pca.transform(d_sift))
  
        #Predict using KMEANS to get bag of words for svm training objects
        clusters = obj_kmeans_total.predict(pca_desc.iloc[:,0:50]) 
        cluster_freq_pivot = desc_freq(clusters)
        
        #Get correct labels
        tree = ET.parse(train_ann_dir + img_name + '.xml')
        root = tree.getroot() 
        
        class_names = []                
        for child in root:
            if child.tag == "object":
                obj_name = child.find("name").text
                if obj_name  not in class_names:
                    class_names += [obj_name]
                    #img_labels += [class_names]
        for class_name in class_names:
                desc = np.append(desc, cluster_freq_pivot, axis=0)
                img_labels += [class_name]
           
desc = desc[1:]

#1 class SVM for each object
svms = {}
svm_acc = {}
for key in Classes:
    #Get x and y for the object type
    X = pd.DataFrame(desc).copy()
    y = pd.DataFrame(np.array(img_labels)).copy()
    #X = normalize(X,axis=1)
    
    #Transform into binary classification for 1 class SVMs
    y[y != key] = '0'
    y[y == key] = '1'
    y = y.astype('int64')

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    #Train and store svm in dictionary
    svm_clf = svm.SVC(kernel='linear').fit(X_train,y_train)
    svms[key] = svm_clf
    
    #Store results
    y_pred = svm_clf.predict(X_test)
    svm_acc[key] = accuracy_score(y_test, y_pred)
    
    print(str(key) + ' svm complete') 
#%%
pickle.dump(svms,open('D:/GD/MLDM/Computer_Vision_Project/svms_obj4.sav','wb'))    
pickle.dump(svm_acc,open('D:/GD/MLDM/Computer_Vision_Project/svm_acc_obj4.sav','wb')) 

#%%
#Load SVM example
svms = pickle.load(open('D:/GD/MLDM/Computer_Vision_Project/svms_obj.sav','rb'))    
svm_acc = pickle.load(open('D:/GD/MLDM/Computer_Vision_Project/svm_acc_obj.sav','rb')) 

#%%
#Get frequency of each descripter
def desc_freq(clusters):
    freq = np.array(np.repeat(0,1000),dtype='float64').reshape((1,1000))
    code, counts = np.unique(clusters, return_counts=True)
    
    for j in range(len(code)):
        freq[0,code[j]] = counts[j]
    
    return freq
#%%
Classes = os.listdir('D:/GD/MLDM/Computer_Vision_Project/cnn5/data/training')
test_dir = 'D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/JPEGImages/'
test_ann_dir = "D:/GD/MLDM/Computer_Vision_Project/Data/test_VOC2007/Annotations2/"



#%%
#Get all test images
test_imgs = os.listdir(test_dir)
test_anns = os.listdir(test_ann_dir)
pred_class = pd.DataFrame(index=Classes)
true_class = pd.DataFrame(index=Classes)

'''
if specific_img:
    test_imgs = [x for x in test_imgs if x in specific_img]
'''

#Iterate and get prediction and correct class tables
desc = []
img_num = []
img_labels = []
for img_name in test_imgs:
    img_name = img_name[:-4]
    img_num = img_num + [img_name]
    #Ensure we have correct label values
    if (img_name+'.xml') in test_anns:
        
        print('descr: ' + img_name)
        #Load image
        #img = Image.open(test_dir+img_name+'.jpg')
        img = cv2.imread((test_dir+img_name+'.jpg'))
    
        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #load feature detection algorithms
        #brisk=cv2.BRISK_create()
        sift = cv2.xfeatures2d.SIFT_create()
  
        #Run feature detection algorithm
        #k_brisk = brisk.detect(gray,None)
        k_sift = sift.detect(gray,None)
       
        #Computer keypoints and descriptors
        #k_brisk,d_brisk = brisk.compute(gray,k_brisk)
        k_sift,d_sift = sift.compute(gray,k_sift)
        desc = desc + [d_sift]

        #Use pca to reduce dimensionality
        pca_desc = pd.DataFrame(pca.transform(d_sift))
  
        #Predict using KMEANS to get bag of words for svm training objects
        clusters = obj_kmeans_total.predict(pca_desc.iloc[:,0:50])          
        
        #Format to frequency of cluster for svm training feed
        #obj_cluster_pivot = km.cluster_format(obj_clusters_total, obj_id,obj_name)
        cluster_freq_pivot = desc_freq(clusters)  
        #Normalize
        cluster_freq_pivot = normalize(cluster_freq_pivot,axis=1)

        #Predict using SVM
        preds = pd.DataFrame()
        for label, svm in svms.items():
                pred = svm.predict(cluster_freq_pivot)
                preds.loc[label,img_name] = pred
        
        #Continually combine
        pred_class = pred_class.join(preds)
        
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
                    img_labels += [class_names]
        
        #Create one hot encoding
        one_hot = pd.DataFrame(np.repeat(0,20),index=Classes,columns=[img_name])
        for class_name in class_names:
            one_hot.loc[class_name,img_name] = 1
        true_class = true_class.join(one_hot)
        
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

#Print prediction vs actual
for x in pred_class.columns:
    print('#######################################')
    print('Image: ' + str(x))
    print('***************************************')
    print('Predicted Labels:')
    for y in pred_class.index:
        print(str(y) + ': ' + str(pred_class.loc[y,x]))                
    print('***************************************')
    print('True Labels:')
    for y in pred_class.index:
        print(str(y) + ': ' + str(true_class.loc[y,x]))                
    print('***************************************')
print('True Positives: ' + str(tp))
print('False Positives: ' + str(fp))
print('True Negatives: ' + str(tn))
print('False Negatives: ' + str(fn))
#Accuracy, precision, recall
print('Accuracy: ' + str((tp+tn)/(tp+fp+tn+fn)))
print('Precision: ' + str(tp/(tp+fp)))
print('Recall: ' + str(tp/(tp+fn)))
print('#######################################')
            
pickle.dump([desc,img_num,img_labels],open('D:/GD/MLDM/Computer_Vision_Project/desc_label_BOW.sav','wb'))    


#%%
###############################################################################
# Other
###############################################################################
results = pd.DataFrame(columns=['tp','fp','tn','fn','acc','prec',
                                'rec'])     
results.loc['seperated','tp'] = tp
results.loc['seperated','fp'] = fp
results.loc['seperated','tn'] = tn
results.loc['seperated','fn'] = fn
results.loc['seperated','acc'] = ((tp+tn)/(tp+fp+tn+fn))
results.loc['seperated','prec'] = (tp/(tp+fp))
results.loc['seperated','rec'] = (tp/(tp+fn))
#%%
results.to_csv('seperated_class_kmeans.csv')