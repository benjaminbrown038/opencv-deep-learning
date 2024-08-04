import numpy as np
import cv2
import sys
import requests
from os import path
import matplotlib.pyplot as plt


modelConfiguration = "yolov4.cfg"
modelWeights = "yolov4.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1 , crop = False)
outs = net.forward(getOutputsNames(net))

tiny_modelConfiguration = "yolov4-tiny.cfg"
tiny_modelWeights = "yolov4-tiny.weights"

net_tiny = cv2.dnn.readNetFromDarknet(tiny_modelConfiguration, tiny_modelWeights)
blob = cv2.dnn.blobFromImage(frame,1/255,(inpWidth,inpHeight), [0,0,0],1,crop = False)
outs = net.forward(getOutputsNames(net_tiny))
