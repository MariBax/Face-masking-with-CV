import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils, translate, resize
from imutils.video import VideoStream, FPS, FileVideoStream
import time

from scipy.spatial import distance as dist
import math

from utils import *
from mask_image import *

RED_EYE_FILE = 'drive/My Drive/Computer Vision/Project/masks/3_red_eye_v2.PNG'
BLUE_EYE_FILE = 'drive/My Drive/Computer Vision/Project/masks/3_blue_eye.PNG'
ANGEL_HAT_FILE = 'drive/My Drive/Computer Vision/Project/masks/3_angel_hat.PNG'
DEVIL_EAR_FILE = 'drive/My Drive/Computer Vision/Project/masks/3_devil_ear.PNG'

class VidCap:
    def __init__(self, video_file):
        self.vidcap = cv2.VideoCapture(video_file)

    def get_frame(self, sec, show=False):
        # get frame at second sec
        # returns 
        # - has_frames : bool
        # - frame      : image or 0

        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000) # takes frame at time sec
        has_frames, image = self.vidcap.read()
        if has_frames:
            if show:
                plt.imshow(image)
                plt.show()
            return has_frames, image
        else:
            return has_frames, 0

    def extract_frames(self, frame_rate=1, max_frames=20, show=False):
        # capture image in each frame_rate seconds
        frames = []

        sec = 0
        count = 1
        has_frames, frame = self.get_frame(sec, show)

        while has_frames and count <= max_frames:
            frames.append(frame)
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            has_frames, frame = self.get_frame(sec, show)

        return frames


def change_eye(thresh, img, state='angel'):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea)

    for cnt in cnts[-2:]:
        x, y, w, h = cv2.boundingRect(cnt)
        s = min(h, w)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if state =='angel':
            eye_im = cv2.imread(BLUE_EYE_FILE, cv2.IMREAD_UNCHANGED)
        else:
            eye_im = cv2.imread(RED_EYE_FILE, cv2.IMREAD_UNCHANGED)
        eye_im = cv2.resize(eye_im, (s, s))
        img = overlay_transparent(img, eye_im, cx - s // 2, cy - s // 2)
    return img       

def add_hat(img, left_brow, right_brow, state='angel'):
    if state =='angel':
        d = abs(right_brow[0][0] - left_brow[-1][0]) # distance btw brows corners
        hat_im = cv2.imread(ANGEL_HAT_FILE, cv2.IMREAD_UNCHANGED)
        hat_im = imutils.resize(hat_im, width=d)
        y = max(0, right_brow[0][1] - d)
        x = right_brow[0][0] 
        img = overlay_transparent(img, hat_im, x, y)
    else:
        d = abs(right_brow[0][0] - right_brow[-1][0])
        left_ear_im = cv2.imread(DEVIL_EAR_FILE, cv2.IMREAD_UNCHANGED)
        left_ear_im = imutils.resize(left_ear_im, width=d)
        y = max(0, right_brow[0][1] - int(2.5 * d))
        x = right_brow[0][0] - int(d * 0.5)
        img = overlay_transparent(img, left_ear_im, x, y)

        right_ear_im = cv2.flip(left_ear_im, 1)
        y = max(0, left_brow[0][1] - int(2.5 * d))
        x = left_brow[0][0] + int(d * 0.5)
        img = overlay_transparent(img, right_ear_im, x, y)
    return img


def mask_video_angel_devil(input_video_file, output_video_file, show_frames=False, rotate=True):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('drive/My Drive/Computer Vision/Project/models/shape_predictor_68_face_landmarks.dat')
	# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	
	vidcap = VidCap(input_video_file)
	frames = vidcap.extract_frames(frame_rate=0.5) # list of frames

	# start and end points' numbers
	(le_start, le_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(re_start, re_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	(leb_start, leb_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
	(reb_start, reb_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

	curr_state = 'angel' # or devil
	first_blink_frame = 1
	result_frames = []
	EYE_AR_THRESH = 0.25

	for frame in frames:
		frame_c = frame.copy()
		if rotate:
			frame_c = cv2.rotate(frame_c, cv2.ROTATE_90_COUNTERCLOCKWISE);
		gray = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)

		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract start and end points
			left_brow = shape[leb_start:leb_end]
			right_brow = shape[reb_start:reb_end]

			frame_c = add_hat(frame_c, left_brow, right_brow, curr_state)

			left_eye = shape[le_start:le_end]
			right_eye = shape[re_start:re_end]
			left_eye_hull = cv2.convexHull(left_eye)
			right_eye_hull = cv2.convexHull(right_eye)

			# calc eyes ratio
			left_EAR = eye_aspect_ratio(left_eye)
			right_EAR = eye_aspect_ratio(right_eye)
			EAR = (left_EAR + right_EAR) / 2.0

			if EAR < EYE_AR_THRESH and first_blink_frame: # blink
				if curr_state == 'angel':
					curr_state = 'devil'
				else:
					curr_state = 'angel' 
				first_blink_frame = 0
			else: # not blink
				first_blink_frame = 1
				mask = np.zeros(frame_c.shape[:2], dtype=np.uint8)
				mask = cv2.fillConvexPoly(mask, left_eye, 255)
				mask = cv2.fillConvexPoly(mask, right_eye, 255)
				eyes = cv2.bitwise_and(frame_c, frame_c, mask=mask)
				mask = (eyes == [0, 0, 0]).all(axis=2)
				eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

				_, thresh = cv2.threshold(eyes_gray, 80, 255, cv2.THRESH_BINARY)
				thresh = cv2.erode(thresh, None, iterations=2)
				thresh = cv2.dilate(thresh, None, iterations=4)
				thresh = cv2.medianBlur(thresh, 3)
				eyes_gray[thresh == 255] = 0

				# mid = (shape[39][0] + shape[42][0]) // 2
				# contouring(eyes_gray[:, 0:mid], mid, frame_c)
				# contouring(eyes_gray[:, mid:], mid, frame_c, True)
				frame_c = change_eye(eyes_gray, frame_c, curr_state)
	        
			result_frames.append(frame_c)
			frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

			if show_frames:
				plt.figure(figsize=(8, 8))
				plt.imshow(frame_c)
				plt.show()

	# save frames to video
	height, width, _ = result_frames[0].shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
	video=cv2.VideoWriter(output_video_file, fourcc, 3, (width, height)) # the 3rd param is changable

	for img in result_frames:
		video.write(img)

	cv2.destroyAllWindows()
	video.release()



def mask_video_simple(input_video_file, output_video_file, MASK_NAME, add_corona_mask=False, show_frames=False, rotate=True, video_stream=False):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('drive/My Drive/Computer Vision/Project/models/shape_predictor_68_face_landmarks.dat')
	# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	if video_stream: # real-time
		vs = VideoStream(src=0).start()
		out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 3, (253, 450), True)

		while True:
			if vs.more():
				break

			frame = vs.read()
			frame = resize(frame, width=450)
			frame_c = frame.copy()
			if rotate:
				frame_c = cv2.rotate(frame_c, cv2.ROTATE_90_COUNTERCLOCKWISE);

			frame_c = mask_image(frame_c, MASK_NAME, add_corona_mask=add_corona_mask)
			result_frames.append(frame_c)
			frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

			if show_frames:
				plt.figure(figsize=(8, 8))
				plt.imshow(frame_c)
				plt.show()

			# out.write(frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

			cv2.destroyAllWindows()
			vs.stop()
			out.release()

	else: # from video	
		vidcap = VidCap(input_video_file)
		frames = vidcap.extract_frames(frame_rate=0.5) # list of frames

		result_frames = []

		for frame in frames:
			frame_c = frame.copy()
			if rotate:
				frame_c = cv2.rotate(frame_c, cv2.ROTATE_90_COUNTERCLOCKWISE);

			frame_c = mask_image(frame_c, MASK_NAME, add_corona_mask=add_corona_mask)
			result_frames.append(frame_c)
			frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)

			if show_frames:
				plt.figure(figsize=(8, 8))
				plt.imshow(frame_c)
				plt.show()

	# save frames to video
	height, width, _ = result_frames[0].shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
	video=cv2.VideoWriter(output_video_file, fourcc, 3, (width, height)) # the 3rd param is changable

	for img in result_frames:
		video.write(img)

	cv2.destroyAllWindows()
	video.release()
