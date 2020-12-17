import cv2
import dlib
import numpy as np
import math
from scipy.spatial import distance as dist


# CONSTANTS

MOUTH_THRESH = 0.9
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
MOUTH_THRESH = 0.9

MASK_INFO = { 'mask1.jpg': {
                  "src_points": np.float32([[400, 480], [800, 480], [600, 600]]), 
                  "dst_points": 'eyes',
                  "transparent": False},
              'mask2.jpg': {
                  "src_points": np.float32([[270, 400], [680, 400], [470, 550]]),
                  "dst_points": 'eyes',
                  "transparent": False},
              'cat_nose.png': {
                  "src_points": np.float32([[500, 400], [450, 500], [550, 500]]),
                  "dst_points": 'nose',
                  "transparent": True},
              '2_new_year_hat.PNG': {
                  "src_points": np.float32([[250, 750], [400, 850], [550, 750]]),
                  "dst_points": 'brows',
                  "transparent": True},
              'hat.png': {
                  "src_points": np.float32([[150, 620], [250, 644], [350, 620]]),
                  "dst_points": 'brows',
                  "transparent": False},
              'moustache.png': {
                  "src_points": np.float32([[200, 215], [290, 0], [400, 215]]),
                  "dst_points": 'moustache',
                  "transparent": False},
              '1_cat_left_ear.PNG': {
                  "src_points": np.float32([[450, 900], [600, 780], [800, 650]]),
                  "dst_points": 'left_ear',
                  "transparent": True},
              'face_mask.jpeg': {
                  "src_points": np.float32([[120, 185], [35, 55], [185, 55]]),
                  "dst_points": 'lips',
                  "transparent": False}}



########## MOUTH UTILS

def get_lips(shape):
    ulip = np.append(shape[48:55], shape[60:65][::-1], axis=0)
    blip = np.append(shape[54:60], [shape[48]], axis=0)
    blip = np.append(blip, [shape[60]], axis=0)
    blip = np.append(blip, shape[64:68][::-1], axis=0)
    return ulip, blip
    
def get_lip_thikness(lip):
    thikness = 0
    for i in [2, 3, 4]:
        distance = math.sqrt((lip[i][0] - lip[12-i][0])**2 +
                             (lip[i][1] - lip[12-i][1])**2)
        thikness += distance
    return thikness / 3

def get_mouth_height(top_lip, bottom_lip):
    height = 0
    for i in [8, 9, 10]:
        distance = math.sqrt((top_lip[i][0] - bottom_lip[18-i][0])**2 + 
                             (top_lip[i][1] - bottom_lip[18-i][1])**2)
        height += distance
    return height / 3

def check_mouth_open(top_lip, bottom_lip):
    top_lip_height = get_lip_thikness(top_lip)
    bottom_lip_height = get_lip_thikness(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)
    return mouth_height > min(top_lip_height, bottom_lip_height) * MOUTH_THRESH


########## EYES UTILS

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calc_eyes_ratio(leftEye, rightEye):
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0