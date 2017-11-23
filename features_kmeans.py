# -*- coding: utf-8 -*-
"""
Using kmeans to create a bag of visual words for SVM to predict
object recognition in images.

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""


import pymysql
import pymysql.cursors
import json

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os

#%%
def load_objects_desc(n_objects):
    # Connect to the database.
    conn = pymysql.connect(db='images_db', user='mldm_gangster', 
                       passwd='$aint3tienne', port=3306,
                       host='mldm-cv-project.cnpjv4qug6jj.us-east-2.rds.amazonaws.com')

    #conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')
    # Query
    sql_get_descriptors = "SELECT * FROM desc_obj d INNER JOIN objects o \
        ON  d.id_obj = o.ID_object;"
    
    #specify number of objects to load
    #n_objects = 100000 
    
    #Load data
    with conn.cursor() as cursor:
        # Execute sql reques-
        cursor.execute(sql_get_descriptors)
        conn.commit()
        
        #iterators and stoarge lists
        it = 0
        list_conca=[]
        obj_name=[]
        obj_id=[]
        
        #Read data from query
        for row in cursor:
            if it < n_objects:
                json_desc = json.loads(row[2])
                list_conca = list_conca + json_desc['sift']
                print('check1')
                obj_name = obj_name + np.repeat(row[5],len(
                        json_desc['sift'])).tolist()
                print('check2')
                obj_id = obj_id + np.repeat([json_desc['obj']],len(
                        json_desc['sift'])).tolist()
                print("Passage n°",it)
                it+=1
            else:
               break
    
    
    #Combine into df
    #df_conca = pd.concat([pd.Series(list_conca),pd.Series(obj_name),
    #                      pd.Series(obj_id)],axis=1)
    return list_conca, obj_id, obj_name
#%%
#Find optimal number of principle components
def pca_test(pca_conca):
    #Create table to store PCA values
    pca_results = pd.DataFrame(columns=['evr','n_comp'])
    #load pca and run
    for n in range(64):
        pca = PCA(n_components=n)
        pca.fit(pca_conca)
        df = pd.DataFrame.from_dict({"evr":[pca.explained_variance_ratio_.sum()],
                "n_comp":[n]})
        pca_results = pd.concat([pca_results,df])
        print(n)
        print(pca.explained_variance_ratio_.sum())
        
    return pca_results


#%%
def kmeans_test(pca_conca, low_k, high_k, step_k):
    #Table to store kmean reults
    kmeans_results = pd.DataFrame(columns=['k','obj_name','distortion','run_time'])
    #distortions = []
    centroids = {}
    
    #Run kmeans for each object
    for obj in pca_conca.obj_name.unique():
        #Only cluster descriptors from this object
        df = pca_conca[pca_conca.obj_name==obj].drop(['obj_name'],axis=1)
        print("subset taken")
        #iterate through number of k
        for k in range(low_k,high_k,step_k):
        #for k in [50]:
            #Track run time
            start=time.time()
            kmeans = KMeans(n_clusters=k).fit(df)
            end=time.time()
            print(obj + ':complete')
            print(end-start)
            results_df = pd.DataFrame.from_dict({"k":[k],"run_time":[(end-start)],
                "distortion":[sum(np.min(cdist(df, kmeans.cluster_centers_, 
                'euclidean'),axis=1))/pca_conca.shape[1]],"obj_name":[obj]})
            kmeans_results = pd.concat([kmeans_results,results_df])
            centroids[obj] = kmeans.cluster_centers_



    #Total time
    print('Total run time:')
    print(kmeans_results.run_time.sum())
    
    return kmeans_results, centroids
''''
#Write results files to csv's
pca_results.to_csv('pca_results.csv')
kmeans_results.to_csv('kmeans_results.csv')

#Centroids
for keys, values in centroids.items():
    file = 'kmeans_centroids' + str(keys)+'.csv'
    df = pd.DataFrame(values)
    df.to_csv(file)

kmeans_results = pd.read_csv('kmeans_results.csv')    
'''  
def plot_kmeans_results(kmeans_results):
    # Plot the elbow
    for i in kmeans_results.obj_name.unique():
        k = kmeans_results[kmeans_results.obj_name == i].k
        distortions = kmeans_results[kmeans_results.obj_name == i].distortion
        plt.plot(k, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title(i)
        plt.show()



#run kmeans for each object
def kmeans_per_object(pca_conca,k):
    #Table to store kmean reults
    kmeans_results = pd.DataFrame(columns=['k','obj_name','distortion','run_time'])
    #distortions = []
    centroids = {}
    
    #Run kmeans for each object
    for obj in pca_conca.obj_name.unique():
        #Only cluster descriptors from this object
        df = pca_conca[pca_conca.obj_name==obj].drop(['obj_name'],axis=1)
        print("subset taken")
        #Track run time
        start=time.time()
        #create and fit kmeans
        kmeans = KMeans(n_clusters=k).fit(df)
        end=time.time()
        print(obj + ':complete')
        print(end-start)
        results_df = pd.DataFrame.from_dict({"k":[k],"run_time":[(end-start)],
            "distortion":[sum(np.min(cdist(df, kmeans.cluster_centers_, 
            'euclidean'),axis=1))/pca_conca.shape[1]],"obj_name":[obj]})
        #save results
        kmeans_results = pd.concat([kmeans_results,results_df])
        centroids[obj] = kmeans.cluster_centers_

    return kmeans_results, centroids
#%%
'''
#Load centroid results for each object
folder="C:/Users/schwi/Google Drive/MLDM/Computer Vision Project/results"
centroids = {}
for filename in os.listdir(folder):
    if filename != 'desktop.ini':
        #Import image
        cent = pd.DataFrame.from_csv(filename)
        obj = filename[16:].replace('.csv','')
        centroids[obj] = cent
'''
#%%
def combine_clusters(centroids):
    #Combine all object centroids into one df
    centroid_total = pd.DataFrame(columns=list(range(len(centroids['aeroplane']))))
    for keys, values in centroids.items():
        centroid_total = pd.concat([centroid_total,pd.DataFrame(values)])
        
    return centroid_total
    
#%%
def cluster_format(centroids_total, obj_id,obj_name):
    #Combine results to feed to SVM
    kmeans_clusters = pd.concat([pd.Series(centroids_total),pd.Series(obj_id),
                                 pd.Series(obj_name)],axis=1)
    kmeans_clusters = kmeans_clusters.rename(columns={0:'cluster',1:'obj_id',
                                                      2:'obj_name'})
        
    #Get count of each cluster for each object
    cluster_pivot = kmeans_clusters.groupby(['obj_id','cluster'],
                                            as_index=False).count()
    cluster_pivot = cluster_pivot.rename(columns={'obj_name': 'count'})
    
    #Pivot df to have each cluster as col feature and each object as observation
    cluster_pivot = cluster_pivot.pivot(index='obj_id',columns='cluster',
                                        values='count')
    
    #Get labels ready to join
    labels = kmeans_clusters[['obj_id','obj_name']].drop_duplicates().set_index(['obj_id'])
    
    #join labels
    cluster_pivot = cluster_pivot.join(labels, how='inner')
    
    #fill nulls with 0
    cluster_pivot = cluster_pivot.fillna(0)

    return cluster_pivot