# Computer Vision Project
## Object detection and classification using bag-of-visual-words approach with K-Means and SVM and CNN
## Master 2 Machine Learning and Data Mining
### by Austin Schwinn, Jérémie Blanchard, and Oussama Bouldjedri.


## The Database
  In order to store efficiently all the informations that we got from XML files descibing every images, or the data obtained thanks to our computation, we decided to created a MySQL Database.
This Database is composed of three different table : 
* images : Stores general informations for every images of the VOC2007
* objects : Stores informations for every objects extracted from images of the VOC2007
* desc_obj : Stores descriptors obtained from each objects
