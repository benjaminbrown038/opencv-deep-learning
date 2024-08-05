import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

config_file = 'models/DenseNet_121.prototxt'
model_file = 'models/DenseNet_121.caffemodel'

model = cv2.dnn.readNet(model=model_file,config = config_file, framework = 'Caffe')
blob = cv2.dnn.blobFromImage(image = tiger_img, scalefactor=0.017,size = (224,224), mean = (104,117,123), swapRB = False, crop = False)

