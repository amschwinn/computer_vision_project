# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:29:52 2017

@author: jerem
Run a KNN to create different common features

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

#%%

# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')
#%%
sql_get_descriptors = "SELECT * FROM desc_obj d INNER JOIN objects o \
    ON  d.id_obj = o.ID_object;"

#%%
with conn.cursor() as cursor:
    cursor.execute(sql_get_descriptors) #We execute our SQL request
    conn.commit()
    
    it = 0
    list_conca=[]
    obj_name=[]
    obj_id=[]

    for row in cursor:
        if it < 100000:
            json_desc = json.loads(row[2])
            list_conca = list_conca + json_desc['sift']
            obj_name = obj_name + np.repeat(row[5],len(
                    json_desc['sift'])).tolist()
            obj_id = obj_id + [json_desc['obj']]
            print("Passage n°",it)
            it+=1
        else:
           break


#Combine into df
#df_conca = pd.concat([pd.Series(list_conca),pd.Series(obj_name),
#                      pd.Series(obj_id)],axis=1)

#%%
#Transform into df for pca
pca_conca = pd.DataFrame(list_conca)

#Create table to store PCA values
pca_results = pd.DataFrame(columns=['evr','n_comp'])

#%%
#load pca and run
for n in range(64):
    pca = PCA(n_components=n)
    pca.fit(pca_conca)
    df = pd.DataFrame.from_dict({"evr":[pca.explained_variance_ratio_.sum()],
            "n_comp":[n]})
    pca_results = pd.concat([pca_results,df])
    print(n)
    print(pca.explained_variance_ratio_.sum())
    
#%%
#Use desired PCA for actual PCA matrix
n_comp = 50
pca = PCA(n_components=n_comp)
pca.fit(pca_conca)
print(pca.explained_variance_ratio_.sum())
pca_conca = pd.DataFrame(pca.transform(pca_conca))
pca_conca['obj_name'] = pd.Series(obj_name)


#Table to store kmean reults
kmeans_results = pd.DataFrame(columns=['k','obj_name','distortion','run_time'])
distortions = []
centroids = {}

#Run kmeans for each object
for obj in pca_conca.obj_name.unique():
    #Only cluster descriptors from this object
    df = pca_conca[pca_conca.obj_name==obj].drop(['obj_name'],axis=1)
    print("subset taken")
    #iterate through number of k
    #for k in range(10,100,10):
    for k in [50]:
        #Track run time
        start=time.time()
        kmeans = KMeans(n_clusters=k).fit(df)
        end=time.time()
        print(obj + ':complete')
        print(end-start)
        results_df = pd.DataFrame.from_dict({"k":[k],"run_time":[(end-start)],
            "distortion":[sum(np.min(cdist(df, kmeans.cluster_centers_, 
            'euclidean'),axis=1))/n_comp],"obj_name":[obj]})
        kmeans_results = pd.concat([kmeans_results,results_df])
        centroids[obj] = kmeans.cluster_centers_



#Total time
print(kmeans_results.run_time.sum())

#Write results files to csv's
pca_results.to_csv('pca_results.csv')
kmeans_results.to_csv('kmeans_results.csv')

#Centroids
centroids = pd.DataFrame.from_dict(centroids)
centroids.to_csv('kmeans_centroids.csv')

#%%
centroids = pd.DataFrame.from_dict(centroids)
#%%
centroids.to_csv('kmeans_centroids.csv')
#%%
for keys, values in centroids.items():
    file = 'kmeans_centroids' + str(keys)+'.csv'
    df = pd.DataFrame(values)
    df.to_csv(file)


#%%
kmeans_results = pd.read_csv('kmeans_results.csv')    
#%%     
# Plot the elbow
for i in kmeans_results.obj_name.unique():
    k = kmeans_results[kmeans_results.obj_name == i].k
    distortions = kmeans_results[kmeans_results.obj_name == i].distortion
    plt.plot(k, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(i)
    plt.show()