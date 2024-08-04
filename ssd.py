import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib

import requests
import os 
from os import path

modelFile = 'ssd_mobilenet_frozen_inference_graph.pb'
configFile = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
classFile = 'coco_class_labels.txt'

net = cv2.dnn.readNetFromTensorflow(modelFile,configFile)
blob = cv2.dnn.blobFromImage(img,1.0,(dim,dim),mean,True)
