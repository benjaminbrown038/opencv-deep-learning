!git clone https://github.com/ultralytics/yolov5

import numpy as np
import cv2
import sys
import requests
from os import path
import matplotlib.pyplot as plt 

indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
outputs = net.forward(output_layers)

modelWeights = "models/yolov5n.onnx"
net = cv2.dnn.readNet(modelWeights)

modelWeights = "models/yolov5s.onnx"
net = cv2.dnn.readNet(modelWeights)

modelWeights = "models/yolov5m.onnx"
net = cv2.dnn.readNet(modelWeights)

modelWeights = "models/yolov5l.onnx"
net = cv2.dnn.readNet(modelWeights)

modelWeights = "models/yolov5x.onnx"
net = cv2.dnn.readNet(modelWeights)
