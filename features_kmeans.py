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
import matplotlib.pyplot as plt
import numpy as np
import time



# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

sql_get_descriptors = "SELECT * FROM `desc_obj`"

with conn.cursor() as cursor:
    cursor.execute(sql_get_descriptors) #We execute our SQL request
    conn.commit()
    
    it = 0
    list_conca=[]
    
    for row in cursor:
        if it < 1000:
            json_desc = json.loads(row[2])
            list_conca = list_conca + json_desc['sift']
            print("Passage n°",it)
            it+=1
        else:
           break
       

    distortions = []
    
    for k in range(10,15):
        #Track run time
        start=time.time()
        kmeans = KMeans(n_clusters=k).fit(list_conca)
        distortions.append(sum(np.min(cdist(list_conca, kmeans.cluster_centers_, 'euclidean'),axis=1))/64)
        #End timer and show run time
        end=time.time()
        print("Done for nb_cluster=",k," - In ", end - start)
        
     
    # Plot the elbow
    plt.plot(range(10,15), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()