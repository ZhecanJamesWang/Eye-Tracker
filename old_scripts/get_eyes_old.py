# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


def get_eye_patch(image, (name, (i, j)), shape):
	clone = image.copy()
	# loop over the subset of facial landmarks, drawing the
	# specific face part
	top_pts = shape[i:j][:4]
	bottom_pts = shape[i:j][3:]
	bottom_pts = np.concatenate((bottom_pts, [shape[i:j][0]]), axis=0)

	# sub_pts = top_pts
	# sub_pts = bottom_pts
	# plot_pts(sub_pts, clone)

	t_y, t_x, b_x, b_y = [], [], [], []
	for pt in top_pts:
		t_x.append(pt[0])
		t_y.append(pt[1])

	for pt in bottom_pts:
		b_x.append(pt[0])
		b_y.append(pt[1])

	# print "top_pts: ", top_pts
	# print "bottom_pts: ", bottom_pts

	x_min, y_min, x_max, y_max = min(t_x), min(t_y), max(b_x), max(b_y)

	# print "x_min, y_min, x_max, y_max: ", x_min, y_min, x_max, y_max
	# cv2.circle(clone, (x_min, y_min), 1, (0, 255, 0), -1)
	# plot_pts([[x_min, y_min]], clone)
	# plot_pts([[x_min, y_min],[x_max, y_max]], clone)

	padding_h = 10
	padding_v = 24

	(x, y, w, h) = (x_min, y_min, x_max - x_min, y_max - y_min)
	# extract the ROI of the face region as a separate image
	# (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
	roi = image[y - padding_v : y + h + padding_v, x - padding_h : x + w + padding_h]
	# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
	# show the particular face part
	# cv2.imshow("ROI", roi)
	# cv2.imshow("Image", clone)
	return roi

def get_left_right_eyes(image):
	predictor_path = "shape_predictor_68_face_landmarks.dat"
	# args["image"] = "images/example_01.jpg"
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	# image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	print "rects: ", rects

	rect = rects[0]
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the face parts individually
	left_eye = face_utils.FACIAL_LANDMARKS_IDXS.items()[3]
	right_eye = face_utils.FACIAL_LANDMARKS_IDXS.items()[4]
	left_eye = get_eye_patch(image, left_eye, shape)
	right_eye = get_eye_patch(image, right_eye, shape)
	return left_eye, right_eye
