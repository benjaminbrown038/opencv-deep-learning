import cv2
import sys
import os 
import matplotlib.pyplot as plt 
from moviepy.editor import VideoFileClip

# Set up tracker.
tracker_types = ['BOOSTING', 'CSRT', 'KCF', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD']

# Change the index to change the tracker type.
tracker_type = tracker_types[0] 

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()  
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
