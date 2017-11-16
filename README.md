# Computer Vision Project
## Master 2 Machine Learning and Data Mining
### by Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.


## Object detection and classification using 2 approaches:
1. bag-of-visual-words approach with K-Means and SVM 
2. CNN

## The Database
  In order to store efficiently all the informations that we got from XML files descibing every images, or the data obtained thanks to our computation, we decided to created a MySQL Database.
This Database is composed of three different table : 
* images : Stores general informations for every images of the VOC2007
* objects : Stores informations for every objects extracted from images of the VOC2007
* desc_obj : Stores descriptors obtained from each objects

The MySQL database is available for download at this link:
https://drive.google.com/open?id=0B-cXl70btb-qQXBoV1pTNHhQaDg
