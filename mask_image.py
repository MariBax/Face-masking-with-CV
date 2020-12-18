import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils, translate, resize
from imutils.video import VideoStream, FPS, FileVideoStream
import time

from scipy.spatial import distance as dist
import math

from utils import *


def mask_image(frame, MASK_NAME, add_corona_mask):
	ear_cnt = cnt = total = 0

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('drive/My Drive/Computer Vision/Project/models/shape_predictor_68_face_landmarks.dat')
	# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
	    frame, ear_cnt, total, cnt = process_rect(frame, gray, predictor, rect, ear_cnt, total, cnt, 
	    	MASK_NAME, add_corona_mask=add_corona_mask)
	return frame

def process_rect(frame, gray, predictor, rect, ear_cnt, total, cnt, MASK_NAME, add_corona_mask=False, blink_info=False):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# extract start and end points
	leftEye, rightEye = shape[42:48], shape[36:42]
	ulip, blip = get_lips(shape)

	# calc eyes ratio
	ear = calc_eyes_ratio(leftEye, rightEye)
	if ear < EYE_AR_THRESH:
		ear_cnt += 1
	else:
		if ear_cnt >= EYE_AR_CONSEC_FRAMES:
			total += 1
		ear_cnt = 0

	if blink_info:
		cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# if mouth is open, put corona mask
	if add_corona_mask:
		# print(check_mouth_open(ulip, blip))
		if check_mouth_open(ulip, blip):
			cnt += 1
			# if cnt >= 10:
			frame = mask_rect(frame, shape, 'face_mask.jpeg')
			cv2.putText(frame, "Put your mask on!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		else:
			cnt = 0

	return mask_rect(frame, shape, MASK_NAME), ear_cnt, total, cnt


def mask_rect(frame, shape, MASK_NAME):
	mask_dir = 'masks'
	face_mask = cv2.imread(f"{mask_dir}/{MASK_NAME}")

	dst_points = {'eyes': np.float32([shape[36], shape[45], shape[30]]),
	              'nose': np.float32([shape[30], shape[50], shape[52]]),
	              'brows': np.float32([shape[19], shape[27], shape[24]]),
	              'moustache': np.float32([shape[49], shape[33], shape[53]]),
	              'left_ear': np.float32([shape[2], shape[1], shape[0]]),
	              'lips': np.float32([shape[8], shape[1], shape[15]]),
	              }
    
	face_mask_small = face_mask
	gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
	if not MASK_INFO[MASK_NAME]['transparent']:
	    ret, mask = cv2.threshold(gray_mask, 230, 255, cv2.THRESH_BINARY_INV)
	else:
	    ret, mask = cv2.threshold(gray_mask, 30, 255, cv2.THRESH_BINARY)
	face_mask2 = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

	rows, cols = frame.shape[:2]
	M = cv2.getAffineTransform(MASK_INFO[MASK_NAME]['src_points'],
	                           dst_points[MASK_INFO[MASK_NAME]['dst_points']])
	dst = cv2.warpAffine(face_mask2, M, (cols,rows))
	dst_mask = cv2.warpAffine(mask, M, (cols,rows))

	gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	ret, mask_dst = cv2.threshold(dst_mask, 230, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask_dst)

	masked_frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
	masked_dst = cv2.bitwise_and(dst, dst, mask=mask_dst)

	frame = cv2.add(masked_frame, masked_dst)

	return frame
