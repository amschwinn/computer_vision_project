# -*- coding: utf-8 -*-
"""
Load full multi label images

By: Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.

MLDM Master's Year 2
Fall Semester 2017
"""

import pymysql
import pymysql.cursors
import json
from sqlalchemy import create_engine
import pandas as pd

#%%
def load_objects_desc(n_objects):
#%%
    # Connect to the database.
    conn = pymysql.connect(db='images_db', user='mldm_gangster', 
                       passwd='$aint3tienne', port=3306,
                       host='mldm-cv-project.cnpjv4qug6jj.us-east-2.rds.amazonaws.com')

    #conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')
    # Query
    '''
    sql_get_descriptors = "SELECT DISTINCT o.num_img,d.desc_json,o.name \
    FROM (select concat(num_img,'.jpg') as 'num_img',desc_json \
    from desc_img) d INNER JOIN objects o ON  d.num_img = o.num_img \
    GROUP BY num_img, desc_json, name;"
    '''
    sql_get_descriptors = "SELECT * \
    FROM (select concat(num_img,'.jpg') as 'num_img',desc_json \
    from desc_img) d INNER JOIN objects o ON  d.num_img = o.num_img;"
    
    #specify number of objects to load
    #n_objects = 100000 
    
    #Load data
    #with conn.cursor() as cursor:
        # Execute sql reques-
    cursor = conn.cursor()
    cursor.execute(sql_get_descriptors)
    conn.commit()
    
#%%
    engine = create_engine("mysql://mldm_gangster:$aint3tienne@mldm-cv-project.cnpjv4qug6jj.us-east-2.rds.amazonaws.com/images_db")
    #load the table into a data frame
    digits = pd.read_sql_table("user_digits",engine)
#%%        
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