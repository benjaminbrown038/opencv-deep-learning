import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from moviepy.editor import VideoFileClip

tracker_dict = dict(
    BOOSTING   = cv2.legacy.TrackerBoosting_create(),
    CSRT       = cv2.legacy.TrackerCSRT_create(),
    KCF        = cv2.legacy.TrackerKCF_create(),
    MEDIANFLOW = cv2.legacy.TrackerMedianFlow_create(),
    MIL        = cv2.legacy.TrackerMIL_create(),
    MOSSE      = cv2.legacy.TrackerMOSSE_create(),
    TLD        = cv2.legacy.TrackerTLD_create(),
)
