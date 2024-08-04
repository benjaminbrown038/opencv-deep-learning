import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

textDetector = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet50.onnx")

inputSize = (640, 640)

# Set threshold for Binary Map creation and polygon detection.
binThresh = 0.3
polyThresh = 0.5

mean = (122.67891434, 116.66876762, 104.00698793)

textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
textDetector.setInputParams(1.0/255, inputSize, mean, True)

textRecognizer = cv2.dnn_TextRecognitionModel("./resources/crnn_cs.onnx")
textRecognizer.setDecodeType("CTC-greedy")
textRecognizer.setVocabulary(vocabulary)
textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)

boxes, confs = textDetector.detect(image)
dst = getPerspectiveTransform(src, dst[, solveMethod])

dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
