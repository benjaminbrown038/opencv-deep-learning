import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
# East model for text-detection
textDetectorEAST = cv2.dnn_TextDetectionModel_EAST("./resources/frozen_east_text_detection.pb")

# Set the Detection Confidence Threshold and NMS threshold
conf_thresh = 0.8
nms_thresh = 0.4

textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)

# DB model for text-detection based on resnet50
textDetectorDB50 = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet50.onnx")
# DB model for text-detection based on resnet18
textDetectorDB18 = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet18.onnx")

# Set threshold for Binary Map creation and polygon detection
bin_thresh = 0.3
poly_thresh = 0.5

mean = (122.67891434, 116.66876762, 104.00698793)

textDetectorDB18.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
textDetectorDB18.setInputParams(1.0/255, inputSize, mean, True)

textDetectorDB50.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
textDetectorDB50.setInputParams(1.0/255, inputSize, mean, True)

