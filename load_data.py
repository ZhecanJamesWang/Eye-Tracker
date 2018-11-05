import numpy as np
import cv2
import os
import glob
from os.path import join
import json
from utility.data_utility import image_normalization, resize, check_dimension, mirror, contrastBrightess, translate
import tensorflow as tf
import sys

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_data_names_columbia(file_name):
	fh = open(file_name, "r")
	img_list = []
	ang_list = []

	for line in fh.readlines():
		# print (line)
		parts = line.split(" ")
		# print ("parts: ", parts)
		path = parts[0] + " " + parts[1]
		theta = int(parts[2])
		alpha = int(parts[3].replace('\n', ''))
		# print ("path: ", path)
		# print ("theta: ", theta)
		# print ("alpha: ", alpha)
		# raise "debug"

		img_list.append(path)
		ang_list.append([theta, alpha])
	return img_list, ang_list

def load_data_mpii(file):
	npzfile = np.load(file)

	left_images = npzfile["left_image"]
	left_poses = npzfile["left_pose"]
	left_gazes = npzfile["left_gaze"]

	right_images = npzfile["right_image"]
	right_poses = npzfile["right_pose"]
	right_gazes = npzfile["right_gaze"]

	return [left_images, left_poses, left_gazes, right_images, right_poses, right_gazes]


# load data directly from the npz file (small dataset, 48k and 5k for train and test)
def load_data_from_npz(file):

	print("Loading dataset from npz file...")
	npzfile = np.load(file)
	train_eye_left = npzfile["train_eye_left"]
	train_eye_right = npzfile["train_eye_right"]
	train_face = npzfile["train_face"]
	train_face_mask = npzfile["train_face_mask"]
	train_y = npzfile["train_y"]
	val_eye_left = npzfile["val_eye_left"]
	val_eye_right = npzfile["val_eye_right"]
	val_face = npzfile["val_face"]
	val_face_mask = npzfile["val_face_mask"]
	val_y = npzfile["val_y"]
	print("Done.")

	return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]


# load a batch with data loaded from the npz file
def load_batch(data, img_ch, img_cols, img_rows):

	# useful for debug
	save_images = False

	# if save images, create the related directory
	img_dir = "images"
	if save_images:
		if not os.path.exists(img_dir):
			os.makedir(img_dir)

	# create batch structures
	left_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
	right_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
	face_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
	face_grid_batch = np.zeros(shape=(data[0].shape[0], 1, 25, 25), dtype=np.float32)
	y_batch = np.zeros((data[0].shape[0], 2), dtype=np.float32)

	# load left eye
	for i, img in enumerate(data[0]):
		img = cv2.resize(img, (img_cols, img_rows))
		if save_images:
			cv2.imwrite(join(img_dir, "left" + str(i) + ".png"), img)
		img = image_normalization(img)
		left_eye_batch[i] = img.transpose(2, 0, 1)

	# load right eye
	for i, img in enumerate(data[1]):
		img = cv2.resize(img, (img_cols, img_rows))
		if save_images:
			cv2.imwrite("images/right" + str(i) + ".png", img)
		img = image_normalization(img)
		right_eye_batch[i] = img.transpose(2, 0, 1)

	# load faces
	for i, img in enumerate(data[2]):
		img = cv2.resize(img, (img_cols, img_rows))
		if save_images:
			cv2.imwrite("images/face" + str(i) + ".png", img)
		img = image_normalization(img)
		face_batch[i] = img.transpose(2, 0, 1)

	# load grid faces
	for i, img in enumerate(data[3]):
		if save_images:
			cv2.imwrite("images/grid" + str(i) + ".png", img)
		face_grid_batch[i] = img.reshape((1, img.shape[0], img.shape[1]))

	# load labels
	for i, labels in enumerate(data[4]):
		y_batch[i] = labels

	return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


# create a list of all names of images in the dataset
def load_data_names(path):

	seq_list = []
	seqs = sorted(glob.glob(join(path, "0*")))

	for seq in seqs:

		file = open(seq, "r")
		content = file.read().splitlines()
		for line in content:
			seq_list.append(line)

	return seq_list


# load a batch given a list of names (all images are loaded)
def load_batch_from_names(names, path, img_ch, img_cols, img_rows):

	save_img = False

	# data structures for batches
	left_eye_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
	right_eye_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
	face_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
	face_grid_batch = np.zeros(shape=(len(names), 1, 25, 25), dtype=np.float32)
	y_batch = np.zeros((len(names), 2), dtype=np.float32)

	for i, img_name in enumerate(names):

		# directory
		dir = img_name[:5]

		# frame name
		frame = img_name[6:]

		# index of the frame inside the sequence
		idx = int(frame[:-4])

		# open json files
		face_file = open(join(path, dir, "appleFace.json"))
		left_file = open(join(path, dir, "appleLeftEye.json"))
		right_file = open(join(path, dir, "appleRightEye.json"))
		dot_file = open(join(path, dir, "dotInfo.json"))
		grid_file = open(join(path, dir, "faceGrid.json"))

		# load json content
		face_json = json.load(face_file)
		left_json = json.load(left_file)
		right_json = json.load(right_file)
		dot_json = json.load(dot_file)
		grid_json = json.load(grid_file)

		# open image
		img = cv2.imread(join(path, dir, "frames", frame))

		# debug stuff
		if img is None:
			print("Error opening image: {}".format(join(path, dir, "frames", frame)))
			continue

		if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
			int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
			int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
			print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
			continue

		# get face
		tl_x_face = int(face_json["X"][idx])
		tl_y_face = int(face_json["Y"][idx])
		w = int(face_json["W"][idx])
		h = int(face_json["H"][idx])
		br_x = tl_x_face + w
		br_y = tl_y_face + h
		face = img[tl_y_face:br_y, tl_x_face:br_x]

		# get left eye
		tl_x = tl_x_face + int(left_json["X"][idx])
		tl_y = tl_y_face + int(left_json["Y"][idx])
		w = int(left_json["W"][idx])
		h = int(left_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		left_eye = img[tl_y:br_y, tl_x:br_x]

		# get right eye
		tl_x = tl_x_face + int(right_json["X"][idx])
		tl_y = tl_y_face + int(right_json["Y"][idx])
		w = int(right_json["W"][idx])
		h = int(right_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		right_eye = img[tl_y:br_y, tl_x:br_x]

		# get face grid (in ch, cols, rows convention)
		face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
		tl_x = int(grid_json["X"][idx])
		tl_y = int(grid_json["Y"][idx])
		w = int(grid_json["W"][idx])
		h = int(grid_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		face_grid[0, tl_y:br_y, tl_x:br_x] = 1

		# get labels
		y_x = dot_json["XCam"][idx]
		y_y = dot_json["YCam"][idx]

		# resize images
		face = cv2.resize(face, (img_cols, img_rows))
		left_eye = cv2.resize(left_eye, (img_cols, img_rows))
		right_eye = cv2.resize(right_eye, (img_cols, img_rows))

		# save images (for debug)
		if save_img:
			cv2.imwrite("images/face.png", face)
			cv2.imwrite("images/right.png", right_eye)
			cv2.imwrite("images/left.png", left_eye)
			cv2.imwrite("images/image.png", img)

		# normalization
		# face = image_normalization(face)
		# left_eye = image_normalization(left_eye)
		# right_eye = image_normalization(right_eye)

		######################################################

		# transpose images
		face = face.transpose(2, 0, 1)
		left_eye = left_eye.transpose(2, 0, 1)
		right_eye = right_eye.transpose(2, 0, 1)

		# check data types
		face = face.astype('float32')
		left_eye = left_eye.astype('float32')
		right_eye = right_eye.astype('float32')

		# add to the related batch
		left_eye_batch[i] = left_eye
		right_eye_batch[i] = right_eye
		face_batch[i] = face
		face_grid_batch[i] = face_grid
		y_batch[i][0] = y_x
		y_batch[i][1] = y_y

	return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch

def load_batch_from_data_columbia(mtcnn_h, data, batch_size, img_ch, img_cols, img_rows):
	save_img = False

	print ("len(data): ", len(data))
	print ("len(data[0]): ", len(data[0]))
	# print (data)

	[img_list, ang_list] = data
	print ("img_list[:3]: ", img_list[:3])
	print ("ang_list[:3]: ", ang_list[:3])

	# data structures for batches
	left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	face_grid_batch = np.zeros(shape=(batch_size, 25, 25), dtype=np.float32)
	y_batch = np.zeros((batch_size, 2), dtype=np.float32)

	# counter for check the size of loading batch
	b = 0


	for index in range(len(img_list)):
		if index % 20 == 0:
			print ("index: ", index)

		img_name = img_list[index]
		angle = ang_list[index]

		# directory
		dir = img_name

		# open image
		img = cv2.imread(dir)
		img = resize(img, 100)

		# if image is null, skip
		if img is None:
			print("Error opening image: {}".format(dir))
			continue
		try:
			result = mtcnn_h.run_mtcnn(img,  if_face = True, if_facemask = True, if_draw = False)
		except Exception as e:
			print (e)
			# cv2.imwrite("test.png", img)
			# raise "debug"
			continue


		[original, draw, face, left_eye, right_eye, face_mask, left_eye_pts, right_eye_pts] = result

		check_dimension(face, if_even = False)
		check_dimension(left_eye, if_even = True)
		check_dimension(right_eye, if_even = True)

		left_eye, right_eye, face, face_mask = resize(left_eye, 64), resize(right_eye, 64), resize(face, 64), resize(face_mask, 25)

		# print (right_eye.shape)
		# print (left_eye.shape)

		[theta, alpha] = angle

		if save_img:
			cv2.imwrite("images/" + dir + "_" + frame + "_face.png", face)
			cv2.imwrite("images/" + dir + "_" + frame + "_right.png", right_eye)
			cv2.imwrite("images/" + dir + "_" + frame + "_left.png", left_eye)
			cv2.imwrite("images/" + dir + "_" + frame + "_faceGrid.png", face_grid)
			cv2.imwrite("images/" + dir + "_" + frame + "_image.png", img)

			print ("face.shape: ", face.shape)
			print ("left_eye.shape: ", left_eye.shape)
			print ("right_eye.shape: ", right_eye.shape)



		# save images (for debug)
		if save_img:
			increase = 3
			y_x, y_y = - int(y_x * increase), int(y_y * increase)
			print (px, py)
			h, w, _ = face.shape
			cx, cy = w/2.0, h/2.0
			cv2.circle(face,(int(cx), int(cy)), 5, (0,0,255), -1)
			cv2.line(face, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)

		# check data types
		face = face.astype('float32')
		left_eye = left_eye.astype('float32')
		right_eye = right_eye.astype('float32')

		# add to the related batch
		left_eye_batch[b] = left_eye
		right_eye_batch[b] = right_eye
		face_batch[b] = face
		face_grid_batch[b] = face_mask
		y_batch[b][0] = - theta
		y_batch[b][1] = alpha

		# increase the size of the current batch
		b += 1

	return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch, y_batch]

def load_batch_from_data(mtcnn_h, names, path, batch_size, img_ch, img_cols, img_rows, total_num, train_start = None, train_end = None):

	save_img = False
	cum_err_num = 0

	# data structures for batches
	left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	face_grid_batch = np.zeros(shape=(batch_size, 25, 25), dtype=np.float32)
	# face_grid_batch = np.zeros(shape=(batch_size, 25*25), dtype=np.float32)

	y_batch = np.zeros((batch_size, 2), dtype=np.float32)

	# counter for check the size of loading batch
	b = 0


	print ("int(train_start),int(train_end: ", int(train_start),int(train_end))
	# for i in range(int(train_start),int(train_end)):
	i = 0
	orig_batch_size = batch_size

	while (i < batch_size and b < orig_batch_size):
		# print ("i: ", i)
		# print ("orig_batch_size: ", orig_batch_size)
		# print ("batch_size: ", batch_size)
		# print("b: ", b)

		index = train_start + i

		if index > total_num - 1:
			break

		# if i % 50 == 0:
		#     print i
		try:
			# lottery
			# i = np.random.randint(0, len(names))
			# get the lucky one
			img_name = names[index]
		except Exception as e:
			# print ("could not find img_name")
			cum_err_num += 1
			i += 1
			batch_size += cum_err_num
			print (e)
			continue

		# directory
		dir = img_name[:5]

		# frame name
		frame = img_name[6:]

		# index of the frame into a sequence
		idx = int(frame[:-4])

		# open image
		img = cv2.imread(join(path, dir, "frames", frame))
		# print ("img.shape: ", img.shape)

		# if image is null, skip
		if img is None:
			print("Error opening image: {}".format(join(path, dir, "frames", frame)))
			cum_err_num += 1
			i += 1
			batch_size += cum_err_num
			continue


		# try:
		# 	# result = mtcnn_h.run_mtcnn(resize_img,  if_face = False, if_facemask = False, if_draw = False)
		# 	result = mtcnn_h.run_mtcnn(img,  if_face = True, if_facemask = True, if_draw = False)
		#
		# 	# [_, _, _, left_eye, right_eye, _, left_eye_pts, right_eye_pts] = result
		# 	[original, draw, face, left_eye, right_eye, face_grid, left_eye_pts, right_eye_pts] = result
		#
		# 	check_dimension(face)
		# 	check_dimension(face_grid, if_last_channel = False)
		# 	face, face_grid = resize(face, 64), resize(face_grid, 25)
		# 	check_dimension(left_eye, if_even=True)
		# 	check_dimension(right_eye, if_even=True)
		# 	left_eye, right_eye = resize(left_eye, 64), resize(right_eye, 64)
		#
		# 	mtcnn_flag = "True"

		# except Exception as e:
		# print "check eyes check_dimension"
		# print e
		mtcnn_flag = "False"


		# open json files
		face_file = open(join(path, dir, "appleFace.json"))
		left_file = open(join(path, dir, "appleLeftEye.json"))
		right_file = open(join(path, dir, "appleRightEye.json"))
		grid_file = open(join(path, dir, "faceGrid.json"))

		# load json content
		face_json = json.load(face_file)
		left_json = json.load(left_file)
		right_json = json.load(right_file)
		grid_json = json.load(grid_file)

		# # if coordinates are negatives, skip (a lot of negative coords!)
		# if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
		#     int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
		#     int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
		#     print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
		#     continue

		# get face
		tl_x_face = int(face_json["X"][idx])
		tl_y_face = int(face_json["Y"][idx])
		w = int(face_json["W"][idx])
		h = int(face_json["H"][idx])
		br_x = tl_x_face + w
		br_y = tl_y_face + h
		face = img[tl_y_face:br_y, tl_x_face:br_x]

		try:
			# print type(face)
			# print face.shape
			check_dimension(face)
			face = resize(face, 64)
		except Exception as e:
			# print "check face check_dimension"
			# print ("face.shape: ", face.shape)
			cum_err_num += 1
			i += 1
			batch_size += cum_err_num
			# print e
			continue

		# get left eye
		tl_x = tl_x_face + int(left_json["X"][idx])
		tl_y = tl_y_face + int(left_json["Y"][idx])
		w = int(left_json["W"][idx])
		h = int(left_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		left_eye = img[tl_y:br_y, tl_x:br_x]

		# get right eye
		tl_x = tl_x_face + int(right_json["X"][idx])
		tl_y = tl_y_face + int(right_json["Y"][idx])
		w = int(right_json["W"][idx])
		h = int(right_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		right_eye = img[tl_y:br_y, tl_x:br_x]

		try:
			check_dimension(left_eye)
			left_eye = resize(left_eye, 64)
			check_dimension(right_eye)
			right_eye = resize(right_eye, 64)
		except Exception as e:
			# print ("check lef and right eye")
			# print ("left_eye.shape: ", left_eye.shape)
			# print ("right_eye.shape: ", right_eye.shape)
			cum_err_num += 1
			i += 1
			batch_size += cum_err_num
			# print e
			continue

		# get face grid (in ch, cols, rows convention)
		face_grid = np.zeros(shape=(25, 25), dtype=np.float32)
		tl_x = int(grid_json["X"][idx])
		tl_y = int(grid_json["Y"][idx])
		w = int(grid_json["W"][idx])
		h = int(grid_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h

		# print ("face_grid: ", face_grid.shape)
		face_grid[tl_y:br_y, tl_x:br_x] = 1

		dot_file = open(join(path, dir, "dotInfo.json"))
		dot_json = json.load(dot_file)
		# get labels
		y_x = dot_json["XCam"][idx]
		y_y = dot_json["YCam"][idx]

		try:
			# print type(right_eye)
			# print type(left_eye)
			# print type(face)
			#
			# print right_eye.shape
			# print left_eye.shape
			# print face.shape

			face = cv2.resize(face, (img_cols, img_rows))
			left_eye = cv2.resize(left_eye, (img_cols, img_rows))
			right_eye = cv2.resize(right_eye, (img_cols, img_rows))
		except Exception as e:
			print ("checking resizing")
			cum_err_num += 1
			i += 1
			batch_size += cum_err_num
			print (e)
			continue

		# save images (for debug)
		if save_img:
			increase = 3
			y_x, y_y = - int(y_x * increase), int(y_y * increase)
			h, w, _ = face.shape
			cx, cy = w/2.0, h/2.0
			cv2.circle(face,(int(cx), int(cy)), 5, (0,0,255), -1)
			cv2.line(face, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)

			cv2.imwrite("images/" + dir + "_" + frame + "_face_" + mtcnn_flag + ".png", face)
			cv2.imwrite("images/" + dir + "_" + frame + "_right_" + mtcnn_flag + ".png", right_eye)
			cv2.imwrite("images/" + dir + "_" + frame + "_left_" + mtcnn_flag + ".png", left_eye)
			cv2.imwrite("images/" + dir + "_" + frame + "_faceGrid_" + mtcnn_flag + ".png", face_grid)
			cv2.imwrite("images/" + dir + "_" + frame + "_image_" + mtcnn_flag + ".png", img)

			print ("face.shape: ", face.shape)
			print ("left_eye.shape: ", left_eye.shape)
			print ("right_eye.shape: ", right_eye.shape)


		# check data types
		face = face.astype('float32')
		left_eye = left_eye.astype('float32')
		right_eye = right_eye.astype('float32')
		face_grid = face_grid.astype('float32')

		face = image_normalization(face)
		left_eye = image_normalization(left_eye)
		right_eye = image_normalization(right_eye)
		face_grid = face_grid.flatten().astype('float32')

		# add to the related batch
		left_eye_batch[b] = left_eye
		right_eye_batch[b] = right_eye
		face_batch[b] = face
		face_grid_batch[b] = face_grid
		y_batch[b][0] = y_x
		y_batch[b][1] = y_y

		# increase the size of the current batch
		b += 1
		i += 1

	return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch, y_batch], cum_err_num


def save_data_to_tfrecord(mtcnn_h, names, path, img_ch, img_cols, img_rows, if_train, file_name = None):
	save_img = False


	generateFunc = ["original", "resize", "mirror", "translate", "brightnessAndContrast"]

	# open the TFRecords file
	writer = tf.python_io.TFRecordWriter(file_name)

	print("len(names): ", len(names))

	length = len(names)
	# length = 1000

	for i in range(length):
		for index in range(len(generateFunc)):

			# if i % 100 == 0:
			print('Train data {}/{}'.format(i, length))
			print ("index: ", index)

			try:
				img_name = names[i]
			except Exception as e:
				print ("get img_name")
				print 1
				print(e)

			# directory
			dir = img_name[:5]

			# frame name
			frame = img_name[6:]

			# index of the frame into a sequence
			idx = int(frame[:-4])

			# open image
			img = cv2.imread(join(path, dir, "frames", frame))
			# print ("img.shape: ", img.shape)

			# if image is null, skip
			if img is None:
				print("Error opening image: {}".format(join(path, dir, "frames", frame)))
				continue

			method = generateFunc[index]
			(w, h, _) = img.shape

			left_eye_noise = 1
			right_eye_noise = 1
			face_noise = 1

			if method == "resize":
				left_eye_noise = np.random.uniform(1.0, 1.5)
				right_eye_noise = np.random.uniform(1.0, 1.5)
				face_noise = np.random.uniform(1.0, 1.5)

			# try:
			# 	# result = mtcnn_h.run_mtcnn(resize_img,  if_face = False, if_facemask = False, if_draw = False)
			# 	result = mtcnn_h.run_mtcnn(img, if_face=True, if_facemask=True, if_draw=False,
			# 							   left_eye_noise = left_eye_noise, right_eye_noise = right_eye_noise,
			# 							   face_noise = face_noise)
			#
			# 	[original, draw, face, left_eye, right_eye, face_grid, left_eye_pts, right_eye_pts] = result
			#
			# 	check_dimension(face)
			# 	check_dimension(face_grid, if_last_channel = False)
			# 	face, face_grid = resize(face, 64), resize(face_grid, 25)
			# 	check_dimension(left_eye, if_even=True)
			# 	check_dimension(right_eye, if_even=True)
			# 	left_eye, right_eye = resize(left_eye, 64), resize(right_eye, 64)
			#
			# 	mtcnn_flag = "True"
			#
			# except Exception as e:
			print "check eyes check_dimension"
			print e
			mtcnn_flag = "False"

			# open json files
			face_file = open(join(path, dir, "appleFace.json"))
			left_file = open(join(path, dir, "appleLeftEye.json"))
			right_file = open(join(path, dir, "appleRightEye.json"))
			grid_file = open(join(path, dir, "faceGrid.json"))

			# load json content
			face_json = json.load(face_file)
			left_json = json.load(left_file)
			right_json = json.load(right_file)
			grid_json = json.load(grid_file)

			# # if coordinates are negatives, skip (a lot of negative coords!)
			# if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
			#     int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
			#     int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
			#     print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
			#     continue

			# get face
			tl_x_face = int(face_json["X"][idx])
			tl_y_face = int(face_json["Y"][idx])
			w = int(face_json["W"][idx] * face_noise)
			h = int(face_json["H"][idx] * face_noise)
			br_x = tl_x_face + w
			br_y = tl_y_face + h
			face = img[tl_y_face:br_y, tl_x_face:br_x]

			try:
				# print type(face)
				# print face.shape
				check_dimension(face)
				face = resize(face, 64)
			except Exception as e:
				print ("check face check_dimension")
				print 2
				print (e)
				continue

			# get left eye
			tl_x = tl_x_face + int(left_json["X"][idx])
			tl_y = tl_y_face + int(left_json["Y"][idx])
			w = int(left_json["W"][idx] * left_eye_noise)
			h = int(left_json["H"][idx] * left_eye_noise)
			br_x = tl_x + w
			br_y = tl_y + h
			left_eye = img[tl_y:br_y, tl_x:br_x]

			# get right eye
			tl_x = tl_x_face + int(right_json["X"][idx])
			tl_y = tl_y_face + int(right_json["Y"][idx])
			w = int(right_json["W"][idx] * right_eye_noise)
			h = int(right_json["H"][idx] * right_eye_noise)
			br_x = tl_x + w
			br_y = tl_y + h
			right_eye = img[tl_y:br_y, tl_x:br_x]

			try:
				check_dimension(left_eye)
				left_eye = resize(left_eye, 64)
				check_dimension(right_eye)
				right_eye = resize(right_eye, 64)
			except Exception as e:
				print ("check left and right eye size")
				print (e)
				continue

			# get face grid (in ch, cols, rows convention)
			face_grid = np.zeros(shape=(25, 25), dtype=np.float32)
			tl_x = int(grid_json["X"][idx])
			tl_y = int(grid_json["Y"][idx])
			w = int(grid_json["W"][idx])
			h = int(grid_json["H"][idx])
			br_x = tl_x + w
			br_y = tl_y + h

			# print ("face_grid: ", face_grid.shape)
			face_grid[tl_y:br_y, tl_x:br_x] = 1


# --------------------------------------------------------------------

			dot_file = open(join(path, dir, "dotInfo.json"))
			dot_json = json.load(dot_file)
			# get labels
			y_x = dot_json["XCam"][idx]
			y_y = dot_json["YCam"][idx]

			if method == "mirror":
				face, left_eye, right_eye, face_grid, y_x, y_y = mirror(face, left_eye, right_eye, face_grid, y_x, y_y, w=w, h=h)

			elif method == "translate":
				face = translate(face, w=w, h=h)
				left_eye = translate(left_eye, w=w, h=h)
				right_eye = translate(right_eye, w=w, h=h)

			elif method == "brightnessAndContrast":
				face = contrastBrightess(face)
				left_eye = contrastBrightess(left_eye)
				right_eye = contrastBrightess(right_eye)

			elif method == "original":
				pass

			# # check data types
			face = face.astype('float32')
			left_eye = left_eye.astype('float32')
			right_eye = right_eye.astype('float32')
			face_grid = face_grid.astype('float32')

			# save images (for debug)
			if save_img:
				increase = 3
				y_x, y_y = - int(y_x * increase), int(y_y * increase)
				h, w, _ = face.shape
				cx, cy = w/2.0, h/2.0
				cv2.circle(face,(int(cx), int(cy)), 5, (0,0,255), -1)
				cv2.line(face, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)

				cv2.imwrite("images/" + dir + "_" + frame + "_face_" + mtcnn_flag + "_" + method + ".png", face)
				cv2.imwrite("images/" + dir + "_" + frame + "_right_" + mtcnn_flag + "_" + method + ".png", right_eye)
				cv2.imwrite("images/" + dir + "_" + frame + "_left_" + mtcnn_flag + "_" + method + ".png", left_eye)
				cv2.imwrite("images/" + dir + "_" + frame + "_faceGrid_" + mtcnn_flag + "_" + method + ".png", face_grid)
				cv2.imwrite("images/" + dir + "_" + frame + "_image_" + mtcnn_flag + "_" + method + ".png", img)

				print("face.shape: ", face.shape)
				print("left_eye.shape: ", left_eye.shape)
				print("right_eye.shape: ", right_eye.shape)
				print("face_grid.shape: ", face_grid.shape)


			# normalization
			face = image_normalization(face)
			left_eye = image_normalization(left_eye)
			right_eye = image_normalization(right_eye)
			face_grid = face_grid.flatten()


			if if_train:
			    # Create a feature
			    feature = {'train/y_x': _float_feature(y_x),
			               'train/y_y': _float_feature(y_y),
			               'train/face': _bytes_feature(tf.compat.as_bytes(face.tostring())),
			               'train/face_grid': _bytes_feature(tf.compat.as_bytes(face_grid.tostring())),
			               'train/left_eye': _bytes_feature(tf.compat.as_bytes(left_eye.tostring())),
			               'train/right_eye': _bytes_feature(tf.compat.as_bytes(right_eye.tostring()))}
			else:
			    feature = {'test/y_x': _float_feature(y_x),
			               'test/y_y': _float_feature(y_y),
			               'test/face': _bytes_feature(tf.compat.as_bytes(face.tostring())),
			               'test/face_grid': _bytes_feature(tf.compat.as_bytes(face_grid.tostring())),
			               'test/left_eye': _bytes_feature(tf.compat.as_bytes(left_eye.tostring())),
			               'test/right_eye': _bytes_feature(tf.compat.as_bytes(right_eye.tostring()))}

			# Create an example protocol buffer
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			# Serialize to string and write on the file
			writer.write(example.SerializeToString())
		# raise "debug"

	writer.close()
	sys.stdout.flush()


def save_data_to_tfrecord_without_face(mtcnn_h, names, path, img_ch, img_cols, img_rows, file_name = None):
	save_img = False

	# open the TFRecords file
	writer = tf.python_io.TFRecordWriter(file_name)

	print("len(names): ", len(names))

	length = len(names)
	# length = 1000

	for i in range(length):

		if i % 100 == 0:
			print('Train data {}/{}'.format(i, length))

		try:
			# lottery
			# i = np.random.randint(0, len(names))
			# get the lucky one
			img_name = names[i]
		except Exception as e:
			print 1
			print(e)

		# directory
		dir = img_name[:5]

		# frame name
		frame = img_name[6:]

		# index of the frame into a sequence
		idx = int(frame[:-4])

		# open json files
		face_file = open(join(path, dir, "appleFace.json"))
		left_file = open(join(path, dir, "appleLeftEye.json"))
		right_file = open(join(path, dir, "appleRightEye.json"))
		grid_file = open(join(path, dir, "faceGrid.json"))

		# load json content
		face_json = json.load(face_file)
		left_json = json.load(left_file)
		right_json = json.load(right_file)
		grid_json = json.load(grid_file)

		# open image
		img = cv2.imread(join(path, dir, "frames", frame))
		# print ("img.shape: ", img.shape)

		# if image is null, skip
		if img is None:
			print("Error opening image: {}".format(join(path, dir, "frames", frame)))
			continue

		# # if coordinates are negatives, skip (a lot of negative coords!)
		# if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
		#     int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
		#     int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
		#     print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
		#     continue

		# get face
		tl_x_face = int(face_json["X"][idx])
		tl_y_face = int(face_json["Y"][idx])
		w = int(face_json["W"][idx])
		h = int(face_json["H"][idx])
		br_x = tl_x_face + w
		br_y = tl_y_face + h
		face = img[tl_y_face:br_y, tl_x_face:br_x]
		try:
			# print type(face)
			# print face.shape
			check_dimension(face)
			face = resize(face, 64)
		except Exception as e:
			# print "check face check_dimension"
			print 2
			print e
			continue

		try:
			# result = mtcnn_h.run_mtcnn(resize_img,  if_face = False, if_facemask = False, if_draw = False)
			result = mtcnn_h.run_mtcnn(img, if_face=True, if_facemask=True, if_draw=False)

			# [_, _, _, left_eye, right_eye, _, left_eye_pts, right_eye_pts] = result
			[original, draw, face, left_eye, right_eye, face_grid, left_eye_pts, right_eye_pts] = result

			check_dimension(left_eye, if_even=True)
			check_dimension(right_eye, if_even=True)
			left_eye, right_eye = resize(left_eye, 64), resize(right_eye, 64)

			mtcnn_flag = "True"

		except Exception as e:
			# print "check eyes check_dimension"
			print 3
			print e
			mtcnn_flag = "False"

			# get left eye
			tl_x = tl_x_face + int(left_json["X"][idx])
			tl_y = tl_y_face + int(left_json["Y"][idx])
			w = int(left_json["W"][idx])
			h = int(left_json["H"][idx])
			br_x = tl_x + w
			br_y = tl_y + h
			left_eye = img[tl_y:br_y, tl_x:br_x]

			# get right eye
			tl_x = tl_x_face + int(right_json["X"][idx])
			tl_y = tl_y_face + int(right_json["Y"][idx])
			w = int(right_json["W"][idx])
			h = int(right_json["H"][idx])
			br_x = tl_x + w
			br_y = tl_y + h
			right_eye = img[tl_y:br_y, tl_x:br_x]
			try:
				check_dimension(left_eye)
				left_eye = resize(left_eye, 64)
				check_dimension(right_eye)
				right_eye = resize(right_eye, 64)
			except Exception as e:
				print 4
				print e
				continue

			# get face grid (in ch, cols, rows convention)
			face_grid = np.zeros(shape=(25, 25), dtype=np.float32)
			tl_x = int(grid_json["X"][idx])
			tl_y = int(grid_json["Y"][idx])
			w = int(grid_json["W"][idx])
			h = int(grid_json["H"][idx])
			br_x = tl_x + w
			br_y = tl_y + h

			# print ("face_grid: ", face_grid.shape)
			face_grid[tl_y:br_y, tl_x:br_x] = 1

		dot_file = open(join(path, dir, "dotInfo.json"))
		dot_json = json.load(dot_file)
		# get labels
		y_x = dot_json["XCam"][idx]
		y_y = dot_json["YCam"][idx]

		print("face.shape: ", face.shape)
		print("left_eye.shape: ", left_eye.shape)
		print("right_eye.shape: ", right_eye.shape)

		try:
			# print type(right_eye)
			# print type(left_eye)
			# print type(face)
			#
			# print right_eye.shape
			# print left_eye.shape
			# print face.shape

			face = cv2.resize(face, (img_cols, img_rows))
			left_eye = cv2.resize(left_eye, (img_cols, img_rows))
			right_eye = cv2.resize(right_eye, (img_cols, img_rows))
		except Exception as e:
			print 5
			print("checking resizing")
			print(e)

		# if save_img:
		cv2.imwrite("images/" + dir + "_" + frame + "_face_" + mtcnn_flag + ".png", face)
		cv2.imwrite("images/" + dir + "_" + frame + "_right_" + mtcnn_flag + ".png", right_eye)
		cv2.imwrite("images/" + dir + "_" + frame + "_left_" + mtcnn_flag + ".png", left_eye)
		cv2.imwrite("images/" + dir + "_" + frame + "_faceGrid_" + mtcnn_flag + ".png", face_grid)
		cv2.imwrite("images/" + dir + "_" + frame + "_image_" + mtcnn_flag + ".png", img)

		# # save images (for debug)
		# if save_img:
		#     increase = 3
		#     y_x, y_y = - int(y_x * increase), int(y_y * increase)
		#     print (px, py)
		#     h, w, _ = face.shape
		#     cx, cy = w/2.0, h/2.0
		#     cv2.circle(face,(int(cx), int(cy)), 5, (0,0,255), -1)
		#     cv2.line(face, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)

		# normalization
		# face = image_normalization(face)
		# left_eye = image_normalization(left_eye)
		# right_eye = image_normalization(right_eye)
		face_grid = face_grid.flatten().astype('float32')

		# check data types
		face = face.astype('float32')
		left_eye = left_eye.astype('float32')
		right_eye = right_eye.astype('float32')
		face_grid = face_grid.astype('float32')


		if file_name == "train.tfrecords":
			# Create a feature
			feature = {'train/y_x': _float_feature(y_x),
					   'train/y_y': _float_feature(y_y),
					   'train/face': _bytes_feature(tf.compat.as_bytes(face.tostring())),
					   'train/face_grid': _bytes_feature(tf.compat.as_bytes(face_grid.tostring())),
					   'train/left_eye': _bytes_feature(tf.compat.as_bytes(left_eye.tostring())),
					   'train/right_eye': _bytes_feature(tf.compat.as_bytes(right_eye.tostring()))}
		else:
			feature = {'test/y_x': _float_feature(y_x),
					   'test/y_y': _float_feature(y_y),
					   'test/face': _bytes_feature(tf.compat.as_bytes(face.tostring())),
					   'test/face_grid': _bytes_feature(tf.compat.as_bytes(face_grid.tostring())),
					   'test/left_eye': _bytes_feature(tf.compat.as_bytes(left_eye.tostring())),
					   'test/right_eye': _bytes_feature(tf.compat.as_bytes(right_eye.tostring()))}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()



# load a batch of random data given the full list of the dataset
def load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows):

	save_img = False

	# data structures for batches
	left_eye_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
	right_eye_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
	face_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
	face_grid_batch = np.zeros(shape=(batch_size, 1, 25, 25), dtype=np.float32)
	y_batch = np.zeros((batch_size, 2), dtype=np.float32)

	# counter for check the size of loading batch
	b = 0

	while b < batch_size:

		# lottery
		i = np.random.randint(0, len(names))

		# get the lucky one
		img_name = names[i]

		# directory
		dir = img_name[:5]

		# frame name
		frame = img_name[6:]

		# index of the frame into a sequence
		idx = int(frame[:-4])

		# open json files
		face_file = open(join(path, dir, "appleFace.json"))
		left_file = open(join(path, dir, "appleLeftEye.json"))
		right_file = open(join(path, dir, "appleRightEye.json"))
		dot_file = open(join(path, dir, "dotInfo.json"))
		grid_file = open(join(path, dir, "faceGrid.json"))

		# load json content
		face_json = json.load(face_file)
		left_json = json.load(left_file)
		right_json = json.load(right_file)
		dot_json = json.load(dot_file)
		grid_json = json.load(grid_file)

		# open image
		img = cv2.imread(join(path, dir, "frames", frame))

		# if image is null, skip
		if img is None:
			# print("Error opening image: {}".format(join(path, dir, "frames", frame)))
			continue

		# if coordinates are negatives, skip (a lot of negative coords!)
		if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
			int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
			int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
			# print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
			continue

		# get face
		tl_x_face = int(face_json["X"][idx])
		tl_y_face = int(face_json["Y"][idx])
		w = int(face_json["W"][idx])
		h = int(face_json["H"][idx])
		br_x = tl_x_face + w
		br_y = tl_y_face + h
		face = img[tl_y_face:br_y, tl_x_face:br_x]

		# get left eye
		tl_x = tl_x_face + int(left_json["X"][idx])
		tl_y = tl_y_face + int(left_json["Y"][idx])
		w = int(left_json["W"][idx])
		h = int(left_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		left_eye = img[tl_y:br_y, tl_x:br_x]

		# get right eye
		tl_x = tl_x_face + int(right_json["X"][idx])
		tl_y = tl_y_face + int(right_json["Y"][idx])
		w = int(right_json["W"][idx])
		h = int(right_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		right_eye = img[tl_y:br_y, tl_x:br_x]

		# get face grid (in ch, cols, rows convention)
		face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
		tl_x = int(grid_json["X"][idx])
		tl_y = int(grid_json["Y"][idx])
		w = int(grid_json["W"][idx])
		h = int(grid_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h
		face_grid[0, tl_y:br_y, tl_x:br_x] = 1

		# get labels
		y_x = dot_json["XCam"][idx]
		y_y = dot_json["YCam"][idx]

		# resize images
		face = cv2.resize(face, (img_cols, img_rows))
		left_eye = cv2.resize(left_eye, (img_cols, img_rows))
		right_eye = cv2.resize(right_eye, (img_cols, img_rows))


		# save images (for debug)
		if save_img:
			cv2.imwrite("images/face.png", face)
			cv2.imwrite("images/right.png", right_eye)
			cv2.imwrite("images/left.png", left_eye)
			cv2.imwrite("images/image.png", img)

		# normalization
		face = image_normalization(face)
		left_eye = image_normalization(left_eye)
		right_eye = image_normalization(right_eye)

		######################################################

		# transpose images
		face = face.transpose(2, 0, 1)
		left_eye = left_eye.transpose(2, 0, 1)
		right_eye = right_eye.transpose(2, 0, 1)

		# check data types
		face = face.astype('float32')
		left_eye = left_eye.astype('float32')
		right_eye = right_eye.astype('float32')

		# add to the related batch
		left_eye_batch[b] = left_eye
		right_eye_batch[b] = right_eye
		face_batch[b] = face
		face_grid_batch[b] = face_grid
		y_batch[b][0] = y_x
		y_batch[b][1] = y_y

		# increase the size of the current batch
		b += 1

	return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


if __name__ == "__main__":

	# debug
	seq_list = load_data_names("/cvgl/group/GazeCapture/test")

	batch_size = len(seq_list)
	dataset_path = "/cvgl/group/GazeCapture/gazecapture"
	img_ch = 3
	img_cols = 64
	img_rows = 64

	test_batch = load_batch_from_names_random(seq_list, dataset_path, batch_size, 3, 64, 64)

	print("Loaded: {} data".format(len(test_batch[0][0])))
