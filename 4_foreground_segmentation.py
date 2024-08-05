import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
results = segment.process(img)
img_seg_mask = results.segmentation_mask
mask3d = np.dstack((binary_mask, binary_mask, binary_mask))
