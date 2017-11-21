# -*- coding: utf-8 -*-
"""
Using kmeans to create a bag of visual words and SVM to predict
object recognition in images.

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

import features_kmeans as km
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.grid_search import GridSearchCV



#%%
#Load object descriptors
obj_desc_list, obj_id, obj_name = km.load_objects_desc(20000)

#%%
###############################################################################
# PCA
###############################################################################
#Transform into df for pca
pca_obj_desc = pd.DataFrame(obj_desc_list)

#Find optimal principle components
#pca_results = km.pca_test(pca_obj_desc)

#Run PCA and transform to optimal number of PCs
n_comp = 50
pca = PCA(n_components=n_comp)
pca.fit(pca_obj_desc)
pca_obj_desc = pd.DataFrame(pca.transform(pca_obj_desc))


#%%
###############################################################################
# K-Means
###############################################################################
#Combine PCA results with obj name
pca_obj_desc['obj_name'] = pd.Series(obj_name)

#Run test kmeans results to find optimal k
#test_kmeans_results, test_centroids = km.kmeans_test(pca_obj_desc, 10, 100, 10)

#Run kmeans on object descriptors to get our visual bag of words for each obj
obj_kmeans_results, obj_centroids = km.kmeans_per_object(pca_obj_desc,50)

#Combine all separate obj centroids into a single group of centroids for our 
#total bag of visual words
tot_obj_centroids = km.combine_clusters(obj_centroids)

#Create k-means model with the all object centroids 
obj_kmeans_total = KMeans(init=tot_obj_centroids,
                          n_clusters=len(tot_obj_centroids),
                          n_init=1).fit(tot_obj_centroids)

#Predict to get bag of words for svm training objects
obj_clusters_total = obj_kmeans_total.predict(pca_obj_desc.iloc[:,0:50])

#Format for svm training feed
obj_cluster_pivot = km.cluster_format(obj_clusters_total, obj_id,obj_name)

#%%
###############################################################################
# SVM
###############################################################################

#split features and labels
X = obj_cluster_pivot.drop(['obj_name'],axis=1)
y = obj_cluster_pivot['obj_name']

#Normalize X
X = normalize(X)

#Create and train SVM model
#svm_clf = svm.SVC().fit(X, y) 

tuned_params = [{'kernel': ['rbf','sigmoid'], 'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                {'kernel': ['poly'], 'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100, 1000], 'degree':[1,2,3,4,5,6,7]}]

scores = ['precision', 'recall','accuracy']


svm_clf = GridSearchCV(svm.SVC(), tuned_params, cv=5)
svm_clf.fit(X, y)






#results = cross_val_score(svm.SVC(),X,y, fit_params={'kernel':['linear','poly','rbf','sigmoid'],'degree':[range(1,10)],})