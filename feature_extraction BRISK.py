
import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#def load_images_from_folder(folder):

kp = [] 
dsc = []
#images = []
folder="ii"    ## you have to set this for your own path
start=time.time()
for filename in os.listdir(folder):
  img = cv2.imread(os.path.join(folder,filename))
  #print("passed")
  brisk=cv2.BRISK_create()

  k=brisk.detect(img,None)

  k,d=brisk.compute(img,k)

  kp.append(k)     
  dsc.append(d)

  #img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
  
  #plt.imshow(img2),plt.show()
   




end=time.time()
print(end - start)

