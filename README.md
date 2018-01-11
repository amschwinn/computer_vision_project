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

## Predicting on Test Set
Use the files Predict_CNN or Predict_BOW to predict using the algorithm of your choice.

To run, edit the parameters section at the top of the script. These include setting the test directory.
Once the parameters section has been edited, simply run the script for predictions.

Dependency serialized objects are available at the following link:
https://drive.google.com/drive/folders/1eLrjXlxiVoJF7eQhZ5qbdeiEp_T2VsBr?usp=sharing

## References
* https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

* https://www.tensorflow.org/tutorials/image\_recognition

* https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2
