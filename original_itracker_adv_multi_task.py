import argparse
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import load_data_names, load_batch_from_data, load_data_mpii, load_data_names_columbia
import datetime
import random
from mtcnn.mtcnn import mtcnn_handle

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

mtcnn_h = mtcnn_handle()
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M")

dataset_path = "..\Eye-Tracking-for-Everyone-master\Eye-Tracking-for-Everyone-master\GazeCapture"
train_path = dataset_path + '\ '.strip() + "train"
val_path = dataset_path + '\ '.strip() + "validation"
test_path = dataset_path + '\ '.strip() + "test"
# dataset_path = "../data/GazeCapture"
# train_path = dataset_path + '/' + "train"
# val_path = dataset_path + '/' + "validation"
# test_path = dataset_path + '/' + "test"

img_cols = 64
img_rows = 64
img_ch = 3


# Network Parameters
img_size = 64
n_channel = 3
mask_size = 25

# pathway: eye_left and eye_right
conv1_eye_size = 11
conv1_eye_out = 96
pool1_eye_size = 2
pool1_eye_stride = 2

conv2_eye_size = 5
conv2_eye_out = 256
pool2_eye_size = 2
pool2_eye_stride = 2

conv3_eye_size = 3
conv3_eye_out = 384
pool3_eye_size = 2
pool3_eye_stride = 2

conv4_eye_size = 1
conv4_eye_out = 64
pool4_eye_size = 2
pool4_eye_stride = 2

eye_size = 2 * 2 * 2 * conv4_eye_out
left_eye_size = 2 * 2 * conv4_eye_out

# pathway: face
conv1_face_size = 11
conv1_face_out = 96
pool1_face_size = 2
pool1_face_stride = 2

conv2_face_size = 5
conv2_face_out = 256
pool2_face_size = 2
pool2_face_stride = 2

conv3_face_size = 3
conv3_face_out = 384
pool3_face_size = 2
pool3_face_stride = 2

conv4_face_size = 1
conv4_face_out = 64
pool4_face_size = 2
pool4_face_stride = 2

face_size = 2 * 2 * conv4_face_out

# fc layer
fc_eye_size = 128
fc_face_size = 128
fc_face_mask_size = 256
face_face_mask_size = 128
fc_size = 128
fc2_size = 2


def convert_to_unit_vector(angles):
	x = -tf.cos(angles[:, 0]) * tf.sin(angles[:, 1])
	y = -tf.sin(angles[:, 0])
	# z = -tf.cos(angles[:, 1]) * tf.cos(angles[:, 1])
	z = -tf.cos(angles[:, 1])
	# * tf.cos(angles[:, 1])

	norm = tf.sqrt(x**2 + y**2 + z**2)
	x /= norm
	y /= norm
	z /= norm
	return x, y, z


def compute_angle_error(labels, preds):
	pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
	label_x, label_y, label_z = convert_to_unit_vector(labels)
	angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
	return tf.reduce_mean(tf.acos(angles) * 180 / np.pi)


def normalize(data):
	shape = data.shape
	data = np.reshape(data, (shape[0], -1))
	data = data.astype('float32') / 255. # scaling
	data = data - np.mean(data, axis=0) # normalizing
	return np.reshape(data, shape)

def prepare_data(data):
	eye_left, eye_right, face, face_mask, y = data
	eye_left = normalize(eye_left)
	eye_right = normalize(eye_right)
	face = normalize(face)
	face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
	y = y.astype('float32')
	return [eye_left, eye_right, face, face_mask, y]

def prepare_data_mpii(data):
	images, gazes = data
	# images = normalize(images)

	y = gazes.astype('float32')

	return [images, y]

def shuffle_data(data):
	idx = np.arange(data[0].shape[0])
	np.random.shuffle(idx)
	for i in range(len(data)):
		data[i] = data[i][idx]
	return data

def remove_left_over(data, batch_size):
	num = len(data)
	left = num%batch_size
	split = num - left
	data = data[:split]
	return data

def next_batch(data, batch_size):
	for i in np.arange(0, data[0].shape[0], batch_size):
		# yield a tuple of the current batched data
		yield [each[i: i + batch_size] for each in data]

def next_batch_universal(data, batch_size, i):
	# yield a tuple of the current batched data
	# yield [each[i: i + batch_size] for each in data]
	output = []
	for each in data:
		output.append(each[i: i + batch_size])
	i += 1
	if 	i + batch_size >= data[0].shape[0]:
		i = 0
	assert (len(output) == 2)

	return output, i

def split_data(args, x, y, split_ratio = 0.85):
	length = len(x)
	split = int(split_ratio * length)

	train_x = x[:split]
	train_y = y[:split]

	val_x = x[split:]
	val_y = y[split:]

	print ("len(train_data): ", len(train_x))
	print ("len(val_data): ", len(val_x))


	train_x = remove_left_over(train_x, args.batch_size)
	train_y = remove_left_over(train_y, args.batch_size)

	val_x = remove_left_over(val_x, args.batch_size)
	val_y = remove_left_over(val_y, args.batch_size)

	print ("len(train_data): ", len(train_x))
	print ("len(val_data): ", len(val_x))

	train_data = [train_x, train_y]
	val_data = [val_x, val_y]

	return train_data, val_data

def organize_data_(args):
	file_name = "data/columbia_data.txt"
	img_list, ang_list = load_data_names_columbia(file_name)
	train_data, val_data = split_data(args, img_list, ang_list, split_ratio = 0.85)
	return train_data, val_data

def organize_data_mpii(args, direction):
	print ("------ organize_data_mpii --------")
	file_name = "data/ptotal_rgb.npz"
	left_images, left_poses, left_gazes, right_images, right_poses, right_gazes = load_data_mpii(file_name)

	if direction == "left":
		x = left_images
		y = left_gazes
	else:
		x = right_images
		y = right_gazes

	print ("len(data): ", len(x))


	train_data, val_data = split_data(args, x, y, split_ratio = 0.85)

	train_data = prepare_data_mpii(train_data)
	val_data = prepare_data_mpii(val_data)
	print (train_data[0][10:30])
	print ("train_data[0].shape: ", train_data[0].shape)
	print ("train_data[1].shape: ", train_data[1].shape)

	print ("train_data[0][0].shape: ", train_data[0][0].shape)

	return train_data, val_data



class EyeTracker(object):
	def __init__(self):
		# tf Graph input
		self.eye_left = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_left')
		self.eye_right = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_right')
		self.face = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='face')
		self.face_mask = tf.placeholder(tf.float32, [None, mask_size * mask_size], name='face_mask')
		self.y = tf.placeholder(tf.float32, [None, 2], name='pos')
		# Store layers weight & bias
		self.weights = {
			'conv1_eye': tf.get_variable('conv1_eye_w', shape=(conv1_eye_size, conv1_eye_size, n_channel, conv1_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv2_eye': tf.get_variable('conv2_eye_w', shape=(conv2_eye_size, conv2_eye_size, conv1_eye_out, conv2_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv3_eye': tf.get_variable('conv3_eye_w', shape=(conv3_eye_size, conv3_eye_size, conv2_eye_out, conv3_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv4_eye': tf.get_variable('conv4_eye_w', shape=(conv4_eye_size, conv4_eye_size, conv3_eye_out, conv4_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv1_face': tf.get_variable('conv1_face_w', shape=(conv1_face_size, conv1_face_size, n_channel, conv1_face_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv2_face': tf.get_variable('conv2_face_w', shape=(conv2_face_size, conv2_face_size, conv1_face_out, conv2_face_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv3_face': tf.get_variable('conv3_face_w', shape=(conv3_face_size, conv3_face_size, conv2_face_out, conv3_face_out), initializer=tf.contrib.layers.xavier_initializer()),
			'conv4_face': tf.get_variable('conv4_face_w', shape=(conv4_face_size, conv4_face_size, conv3_face_out, conv4_face_out), initializer=tf.contrib.layers.xavier_initializer()),
			'fc_eye': tf.get_variable('fc_eye_w', shape=(eye_size, fc_eye_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc_eye_single': tf.get_variable('fc_eye_w_single', shape=(left_eye_size, fc_eye_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc_face': tf.get_variable('fc_face_w', shape=(face_size, fc_face_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc_face_mask': tf.get_variable('fc_face_mask_w', shape=(mask_size * mask_size, fc_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
			'face_face_mask': tf.get_variable('face_face_mask_w', shape=(fc_face_size + fc_face_mask_size, face_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc': tf.get_variable('fc_w', shape=(fc_eye_size + face_face_mask_size, fc_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc_': tf.get_variable('fc_w_', shape=(fc_eye_size + face_face_mask_size, fc_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc2': tf.get_variable('fc2_w', shape=(fc_size, fc2_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc2_angle_eye': tf.get_variable('fc2_w_angle_eye', shape=(fc_size, fc2_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc2_angle_': tf.get_variable('fc2_w_angle_', shape=(fc_size, fc2_size), initializer=tf.contrib.layers.xavier_initializer())
		}
		self.biases = {
			'conv1_eye': tf.Variable(tf.constant(0.1, shape=[conv1_eye_out])),
			'conv2_eye': tf.Variable(tf.constant(0.1, shape=[conv2_eye_out])),
			'conv3_eye': tf.Variable(tf.constant(0.1, shape=[conv3_eye_out])),
			'conv4_eye': tf.Variable(tf.constant(0.1, shape=[conv4_eye_out])),
			'conv1_face': tf.Variable(tf.constant(0.1, shape=[conv1_face_out])),
			'conv2_face': tf.Variable(tf.constant(0.1, shape=[conv2_face_out])),
			'conv3_face': tf.Variable(tf.constant(0.1, shape=[conv3_face_out])),
			'conv4_face': tf.Variable(tf.constant(0.1, shape=[conv4_face_out])),
			'fc_eye': tf.Variable(tf.constant(0.1, shape=[fc_eye_size])),
			'fc_eye_single': tf.Variable(tf.constant(0.1, shape=[fc_eye_size])),
			'fc_face': tf.Variable(tf.constant(0.1, shape=[fc_face_size])),
			'fc_face_mask': tf.Variable(tf.constant(0.1, shape=[fc_face_mask_size])),
			'face_face_mask': tf.Variable(tf.constant(0.1, shape=[face_face_mask_size])),
			'fc': tf.Variable(tf.constant(0.1, shape=[fc_size])),
			'fc_': tf.Variable(tf.constant(0.1, shape=[fc_size])),
			'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_size])),
			'fc2_angle_eye': tf.Variable(tf.constant(0.1, shape=[fc2_size])),
			'fc2_angle_': tf.Variable(tf.constant(0.1, shape=[fc2_size]))
		}

		# Construct model
		self.pred = self.itracker_nets(self.eye_left, self.eye_right, self.face, self.face_mask, self.weights, self.biases)

	# Create some wrappers for simplicity
	def conv2d(self, x, W, b, strides=1):
		# Conv2D wrapper, with bias and relu activation
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)

	def maxpool2d(self, x, k, strides):
		# MaxPool2D wrapper
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
							  padding='VALID')

	# Create model
	def itracker_nets(self, eye_left, eye_right, face, face_mask, weights, biases):
		# pathway: left eye
		eye_left = self.conv2d(eye_left, weights['conv1_eye'], biases['conv1_eye'], strides=1)
		eye_left = self.maxpool2d(eye_left, k=pool1_eye_size, strides=pool1_eye_stride)

		eye_left = self.conv2d(eye_left, weights['conv2_eye'], biases['conv2_eye'], strides=1)
		eye_left = self.maxpool2d(eye_left, k=pool2_eye_size, strides=pool2_eye_stride)

		eye_left = self.conv2d(eye_left, weights['conv3_eye'], biases['conv3_eye'], strides=1)
		eye_left = self.maxpool2d(eye_left, k=pool3_eye_size, strides=pool3_eye_stride)

		eye_left = self.conv2d(eye_left, weights['conv4_eye'], biases['conv4_eye'], strides=1)
		eye_left = self.maxpool2d(eye_left, k=pool4_eye_size, strides=pool4_eye_stride)

		# pathway: right eye
		eye_right = self.conv2d(eye_right, weights['conv1_eye'], biases['conv1_eye'], strides=1)
		eye_right = self.maxpool2d(eye_right, k=pool1_eye_size, strides=pool1_eye_stride)

		eye_right = self.conv2d(eye_right, weights['conv2_eye'], biases['conv2_eye'], strides=1)
		eye_right = self.maxpool2d(eye_right, k=pool2_eye_size, strides=pool2_eye_stride)

		eye_right = self.conv2d(eye_right, weights['conv3_eye'], biases['conv3_eye'], strides=1)
		eye_right = self.maxpool2d(eye_right, k=pool3_eye_size, strides=pool3_eye_stride)

		eye_right = self.conv2d(eye_right, weights['conv4_eye'], biases['conv4_eye'], strides=1)
		eye_right = self.maxpool2d(eye_right, k=pool4_eye_size, strides=pool4_eye_stride)

		# pathway: face
		face = self.conv2d(face, weights['conv1_face'], biases['conv1_face'], strides=1)
		face = self.maxpool2d(face, k=pool1_face_size, strides=pool1_face_stride)

		face = self.conv2d(face, weights['conv2_face'], biases['conv2_face'], strides=1)
		face = self.maxpool2d(face, k=pool2_face_size, strides=pool2_face_stride)

		face = self.conv2d(face, weights['conv3_face'], biases['conv3_face'], strides=1)
		face = self.maxpool2d(face, k=pool3_face_size, strides=pool3_face_stride)

		face = self.conv2d(face, weights['conv4_face'], biases['conv4_face'], strides=1)
		face = self.maxpool2d(face, k=pool4_face_size, strides=pool4_face_stride)

		# fc layer
		# eye
		eye_left = tf.reshape(eye_left, [-1, int(np.prod(eye_left.get_shape()[1:]))])
		eye_right = tf.reshape(eye_right, [-1, int(np.prod(eye_right.get_shape()[1:]))])
		eye = tf.concat([eye_left, eye_right], 1)
		eye = tf.nn.relu(tf.add(tf.matmul(eye, weights['fc_eye']), biases['fc_eye']))

		eye_left = tf.nn.relu(tf.add(tf.matmul(eye_left, weights['fc_eye_single']), biases['fc_eye_single']))
		eye_right = tf.nn.relu(tf.add(tf.matmul(eye_right, weights['fc_eye_single']), biases['fc_eye_single']))


		# face
		face = tf.reshape(face, [-1, int(np.prod(face.get_shape()[1:]))])
		face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc_face']), biases['fc_face']))

		# face mask
		face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc_face_mask']), biases['fc_face_mask']))

		face_face_mask = tf.concat([face, face_mask], 1)
		face_face_mask = tf.nn.relu(tf.add(tf.matmul(face_face_mask, weights['face_face_mask']), biases['face_face_mask']))

		# all
		fc_in = tf.concat([eye, face_face_mask], 1)
		fc = tf.nn.relu(tf.add(tf.matmul(fc_in, weights['fc']), biases['fc']))
		fc_ = tf.nn.relu(tf.add(tf.matmul(fc_in, weights['fc_']), biases['fc_']))

		out_xy = tf.add(tf.matmul(fc, weights['fc2']), biases['fc2'])

		out_ang_eye_left = tf.add(tf.matmul(eye_left, weights['fc2_angle_eye']), biases['fc2_angle_eye'])
		out_ang_eye_right = tf.add(tf.matmul(eye_right, weights['fc2_angle_eye']), biases['fc2_angle_eye'])

		out_ang_ = tf.add(tf.matmul(fc_, weights['fc2_angle_']), biases['fc2_angle_'])

		return out_xy, out_ang_eye_left, out_ang_eye_right, out_ang_


	def train(self, args, ckpt, plot_ckpt, lr=1e-3, batch_size=128, max_epoch=1000, min_delta=1e-4, patience=10, print_per_epoch=10):
		ifCheck = False

		train_data_, val_data_ = organize_data_(args)
		# --------------------------
		train_data_eye_left, val_data_eye_left = organize_data_mpii(args, "left")
		train_data_eye_right, val_data_eye_right = organize_data_mpii(args, "right")

		# -----------------------------
		print ("------ processing extra data 1 --------")

		train_names = load_data_names(train_path)
		val_names = load_data_names(val_path)

		train_num = len(train_names)
		val_num = len(val_names)

		print ("train_num: ", train_num)
		print ("test_num: ", val_num)

		MaxIters = train_num/batch_size
		n_batches = MaxIters

		val_chunk_size = 1000
		MaxTestIters = val_num/val_chunk_size
		val_n_batches = val_chunk_size/batch_size

		print ("MaxIters: ", MaxIters)
		print ("MaxTestIters: ", MaxTestIters)

		print ('Train on %s samples, validate on %s samples' % (train_num, val_num))


		# Define loss and optimizer
		pred_xy, pred_ang_eye_left,  pred_ang_eye_right, pred_ang_ = self.pred
		self.cost1 = tf.losses.mean_squared_error(self.y, pred_xy)
		self.cost2 = tf.losses.mean_squared_error(self.y, pred_ang_eye_left)
		self.cost3 = tf.losses.mean_squared_error(self.y, pred_ang_eye_right)
		self.cost4 = tf.losses.mean_squared_error(self.y, pred_ang_columbia)



		self.optimizer1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost1)
		self.optimizer2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost2)
		self.optimizer3 = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost3)
		self.optimizer4 = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost4)


		# Evaluate model
		self.err1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred_xy, self.y), axis=1)))
		self.err2 = compute_angle_error(self.y, pred_ang_eye_left)
		self.err3 = compute_angle_error(self.y, pred_ang_eye_right)
		self.err4 = compute_angle_error(self.y, pred_ang_columbia)


		train_loss_history = []
		train_err_history = []
		val_loss_history = []
		val_err_history = []

		train_loss_history_eye_left = []
		train_err_history_eye_left = []
		val_loss_history_eye_left = []
		val_err_history_eye_left = []

		train_loss_history_eye_right = []
		train_err_history_eye_right = []
		val_loss_history_eye_right = []
		val_err_history_eye_right = []

		train_loss_history_columbia = []
		train_err_history_columbia = []
		val_loss_history_columbia = []
		val_err_history_columbia = []

		# n_incr_error = 0  # nb. of consecutive increase in error
		best_loss = np.Inf

		# n_batches = train_data[0].shape[0] / batch_size + (train_data[0].shape[0] % batch_size != 0)

		# Create the collection
		tf.get_collection("validation_nodes")

		# Add stuff to the collection.
		tf.add_to_collection("validation_nodes", self.eye_left)
		tf.add_to_collection("validation_nodes", self.eye_right)
		tf.add_to_collection("validation_nodes", self.face)
		tf.add_to_collection("validation_nodes", self.face_mask)
		tf.add_to_collection("validation_nodes", self.pred)


		# variables_to_restore = [var for var in tf.global_variables()]

		# saver = tf.train.Saver(variables_to_restore)

		saver = tf.train.Saver(max_to_keep = 0)

		# Initializing the variables
		init = tf.global_variables_initializer()
		 # TODO://////
		# tf.reset_default_graph()
		# Launch the graph

		with tf.Session() as sess:
			sess.run(init)
			 # TODO://////
			writer = tf.summary.FileWriter("logs", sess.graph)

			# saver.restore(sess, "./my_model/pretrained/model_4_1800_train_error_3.5047762_val_error_5.765135765075684")
			# saver.restore(sess, "./my_model/2018-09-07-11-15/model_4_420_train_error_2.2030365_val_error_1.8307928442955017")

			# Keep training until reach max iterations
			for n_epoch in range(1, max_epoch + 1):
				print ("vvvvvvvvvvvvvvvvvvv")
				print ("n_epoch: ", n_epoch)
				epoch_start = timeit.default_timer()
				iter_start = None
				# n_incr_error += 1


				# train_names = shuffle_data(train_names)
				random.shuffle(train_names)

				iterTest=0
				i_left = 0
				i_right = 0
				i_ = 0

				for iter in range (int(MaxIters)):

					start = timeit.default_timer()

					# print ("--------------------------------")
					# print ("iter: ", iter)
					train_start=iter * batch_size
					train_end = (iter+1) * batch_size

					batch_train_data_ = next_batch_universal(train_data_, batch_size, i_)
					batch_train_data_ = load_batch_from_data_(mtcnn_h, batch_train_data_, batch_size, img_ch, img_cols, img_rows)
					batch_train_data_ = prepare_data(batch_train_data_)

					batch_train_data = load_batch_from_data(mtcnn_h, train_names, dataset_path, batch_size, img_ch, img_cols, img_rows, train_start = train_start, train_end = train_end)
					batch_train_data = prepare_data(batch_train_data)



					print ('Loading and preparing training data: %.1fs' % (timeit.default_timer() - start))
					start = timeit.default_timer()

					# Run optimization op (backprop)
					sess.run(self.optimizer1, feed_dict={self.eye_left: batch_train_data[0], \
								self.eye_right: batch_train_data[1], self.face: batch_train_data[2], \
								self.face_mask: batch_train_data[3], self.y: batch_train_data[4]})

					train_batch_loss, train_batch_err = sess.run([self.cost1, self.err1], feed_dict={self.eye_left: batch_train_data[0], \
								self.eye_right: batch_train_data[1], self.face: batch_train_data[2], \
								self.face_mask: batch_train_data[3], self.y: batch_train_data[4]})


					# for time in range(5):
					batch_train_data_eye_left, i_left = next_batch_universal(train_data_eye_left, batch_size, i_left)

					sess.run(self.optimizer2, feed_dict={self.eye_left: batch_train_data_eye_left[0], \
								self.y: batch_train_data_eye_left[1]})

					train_batch_loss_eye_left, train_batch_err_eye_left = sess.run([self.cost2, self.err2], feed_dict={self.eye_left: batch_train_data_eye_left[0], \
								self.y: batch_train_data_eye_left[1]})

					batch_train_data_eye_right, i_right = next_batch_universal(train_data_eye_right, batch_size, i_right)

					sess.run(self.optimizer3, feed_dict={self.eye_right: batch_train_data_eye_right[0], \
								self.y: batch_train_data_eye_right[1]})

					train_batch_loss_eye_right, train_batch_err_eye_right = sess.run([self.cost3, self.err3], feed_dict={self.eye_right: batch_train_data_eye_right[0], \
								self.y: batch_train_data_eye_right[1]})

					batch_train_data_columbia, i_columbia = next_batch_universal(train_data_columbia, batch_size, i_columbia)

					# Run optimization op (backprop)
					sess.run(self.optimizer4, feed_dict={self.eye_left: batch_train_data_columbia[0], \
								self.eye_right: batch_train_data_columbia[1], self.face: batch_train_data_columbia[2], \
								self.face_mask: batch_train_data_columbia[3], self.y: batch_train_data_columbia[4]})

					train_batch_loss_columbia, train_batch_err_columbia = sess.run([self.cost4, self.err4], feed_dict={self.eye_left: batch_train_data_columbia[0], \
								self.eye_right: batch_train_data_columbia[1], self.face: batch_train_data_columbia[2], \
								self.face_mask: batch_train_data_columbia[3], self.y: batch_train_data_columbia[4]})


					# print ("train_batch_loss: ", train_batch_loss, "train_batch_err: ", train_batch_err)
					# print ("train_batch_loss_eye: ", train_batch_loss_eye, "train_batch_err_eye: ", train_batch_err_eye)

					train_loss_history.append(train_batch_loss)
					train_err_history.append(train_batch_err)

					train_loss_history_eye_left.append(train_batch_loss_eye_left)
					train_err_history_eye_left.append(train_batch_err_eye_left)

					train_loss_history_eye_right.append(train_batch_loss_eye_right)
					train_err_history_eye_right.append(train_batch_err_eye_right)

					train_loss_history_columbia.append(train_batch_loss_columbia)
					train_err_history_columbia.append(train_batch_err_columbia)

					print ('Training on batch: %.1fs' % (timeit.default_timer() - start))

					# if iter > 1000:
					# 	if iter % 60 == 0:
					# 		ifCheck = True
					# elif iter > 500:
					if iter % 30 == 0:
						ifCheck = True
					# elif iter > 200:
					# 	if iter % 15 == 0:
					# 		ifCheck = True
					# else:
					# 	if iter % 5 == 0:
					# 		ifCheck = True

					if ifCheck:

						start = timeit.default_timer()

						if 	iterTest + 1 >= MaxTestIters:
							iterTest = 0

						test_start=iterTest * val_chunk_size
						test_end = (iterTest+1) * val_chunk_size

						val_data = load_batch_from_data(mtcnn_h, val_names, dataset_path, val_chunk_size, img_ch, img_cols, img_rows, train_start = test_start, train_end = test_end)

						val_n_batches = val_data[0].shape[0] / batch_size + (val_data[0].shape[0] % batch_size != 0)

						val_data = prepare_data(val_data)

						print ('Loading and preparing val data: %.1fs' % (timeit.default_timer() - start))
						start = timeit.default_timer()

						val_loss = 0.
						val_err = 0.
						val_loss_eye_left = 0.
						val_err_eye_left = 0.
						val_loss_eye_right = 0.
						val_err_eye_right = 0.
						val_loss_columbia = 0.
						val_err_columbia = 0.

						i_val_left = 0
						i_val_right = 0
						i_val_columbia  = 0

						for batch_val_data in next_batch(val_data, batch_size):
							batch_val_data_eye_left, i_val_left = next_batch_universal(val_data_eye_left, batch_size, i_val_left)
							batch_val_data_eye_right, i_val_right = next_batch_universal(val_data_eye_right, batch_size, i_val_right)
							batch_val_data_columbia, i_val_columbia = next_batch_universal(val_data_eye_right, batch_size, i_val_right)

							val_batch_loss, val_batch_err = sess.run([self.cost1, self.err1], feed_dict={self.eye_left: batch_val_data[0], \
											self.eye_right: batch_val_data[1], self.face: batch_val_data[2], \
											self.face_mask: batch_val_data[3], self.y: batch_val_data[4]})

							val_batch_loss_eye_left, val_batch_err_eye_left = sess.run([self.cost2, self.err2], \
											feed_dict={self.eye_left: batch_val_data_eye_left[0], \
											self.y: batch_val_data_eye_left[1]})

							val_batch_loss_eye_right, val_batch_err_eye_right = sess.run([self.cost3, self.err3], \
											feed_dict={self.eye_right: batch_val_data_eye_right[0], \
											self.y: batch_val_data_eye_right[1]})

							val_batch_loss_columbia, val_batch_err_columbia = sess.run([self.cost4, self.err4], feed_dict={self.eye_left: batch_val_data_columbia[0], \
											self.eye_right: batch_val_data_columbia[1], self.face: batch_val_data_columbia[2], \
											self.face_mask: batch_val_data_columbia[3], self.y: batch_val_data_columbia[4]})

							val_loss += val_batch_loss / val_n_batches
							val_err += val_batch_err / val_n_batches
							val_loss_eye_left += val_batch_loss_eye_left / val_n_batches
							val_err_eye_left += val_batch_err_eye_left / val_n_batches
							val_loss_eye_right += val_batch_loss_eye_right / val_n_batches
							val_err_eye_right += val_batch_err_eye_right / val_n_batches
							val_loss_columbia += val_batch_loss_columbia / val_n_batches
							val_err_columbia += val_batch_err_columbia/ val_n_batches


						print ("val_loss: ", val_loss, "val_err: ", val_err)
						print ("val_loss_left: ", val_loss_eye_left, "val_err_left: ", val_err_eye_left)
						print ("val_loss_right: ", val_loss_eye_right, "val_err_right: ", val_err_eye_right)


						iterTest += 1

						print ('Testing on chunk: %.1fs' % (timeit.default_timer() - start))
						start = timeit.default_timer()

						if iter_start:
							print ('batch iters runtime: %.1fs' % (timeit.default_timer() - iter_start))
						else:
							iter_start = timeit.default_timer()

						print （"now: ", now）
						print （"learning rate: ", lr）

						print ('Epoch %s/%s Iter %s, train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f'%(n_epoch, max_epoch, iter, np.mean(train_loss_history), np.mean(train_err_history), np.mean(val_loss_history), np.mean(val_err_history)))

						print ('Epoch %s/%s Iter %s, train val_loss_eye_left: %.5f, train error_eye_left: %.5f, val loss_eye_left: %.5f, val error_eye_left: %.5f'%(n_epoch, max_epoch, iter, np.mean(train_loss_history_eye_left), np.mean(train_err_history_eye_left), np.mean(val_loss_history_eye_left), np.mean(val_err_history_eye_left)))

						print ('Epoch %s/%s Iter %s, train loss_eye_right: %.5f, train error_eye_right: %.5f, val loss_eye_right: %.5f, val error_eye_right: %.5f'%(n_epoch, max_epoch, iter, np.mean(train_loss_history_eye_right), np.mean(train_err_history_eye_right), np.mean(val_loss_history_eye_right), np.mean(val_err_history_eye_right)))

						print ('Epoch %s/%s Iter %s, train loss_columbia: %.5f, train error_columbia: %.5f, val loss_columbia: %.5f, val error_columbia: %.5f'%(n_epoch, max_epoch, iter, np.mean(train_loss_history_columbia), np.mean(train_err_history_columbia), np.mean(val_loss_history_columbia), np.mean(val_err_history_columbia)))


						val_loss_history.append(val_loss)
						val_err_history.append(val_err)

						val_loss_history_eye_left.append(val_loss_eye_left)
						val_err_history_eye_left.append(val_err_eye_left)

						val_loss_history_eye_right.append(val_loss_eye_right)
						val_err_history_eye_right.append(val_err_eye_right)

						val_loss_history_columbia.append(val_loss_columbia)
						val_err_history_columbia.append(val_err_columbia)

						plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_loss_history), np.array(val_err_history), start=0, per=1, save_file=plot_ckpt + "/cumul_loss_" + str(n_epoch) + "_" + str(iter) + ".png")

						plot_loss(np.array(train_loss_history_eye_left), np.array(train_err_history_eye_left), np.array(val_loss_history_eye_left), np.array(val_err_history_eye_left), start=0, per=1, save_file=plot_ckpt + "/cumul_loss_" + str(n_epoch) + "_" + str(iter) + "_eye_left.png")

						plot_loss(np.array(train_loss_history_eye_right), np.array(train_err_history_eye_right), np.array(val_loss_history_eye_right), np.array(val_err_history_eye_right), start=0, per=1, save_file=plot_ckpt + "/cumul_loss_" + str(n_epoch) + "_" + str(iter) + "_eye_right.png")

						plot_loss(np.array(train_loss_history_columbia), np.array(train_err_history_columbia), np.array(val_loss_history_columbia), np.array(val_err_history_columbia), start=0, per=1, save_file=plot_ckpt + "/cumul_loss_" + str(n_epoch) + "_" + str(iter) + "_columbia.png")

						# if val_loss - min_delta < best_loss:
						# if val_err - min_delta < best_loss:
							# best_loss = val_err
						save_path = ckpt + "model_" + str(n_epoch) + "_" + str(iter) + "_train_error_%s"%(np.mean(train_err)) + "_val_error_%s"%(np.mean(val_err))

						# saver = tf.train.Saver(max_to_keep=0)
						# , global_step=n_epoch
						save_path = saver.save(sess, save_path)

						print ("Model saved in file: %s" % save_path)
						# n_incr_error = 0

						ifCheck = False

						print ('Saving models and plotting loss: %.1fs' % (timeit.default_timer() - start))


				print ('epoch runtime: %.1fs' % (timeit.default_timer() - epoch_start))

				# train_loss_history.append(np.mean(train_loss))
				# train_err_history.append(np.mean(train_err))
				# val_loss_history.append(np.mean(Val_loss))
				# val_err_history.append(np.mean(Val_err))

				# plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_err_history), start=0, per=1, save_file=plot_ckpt + "/cumul_loss_" + str(n_epoch) + ".png")

				# if n_epoch % print_per_epoch == 0:
				print ('Epoch %s/%s Iter %s, train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f'%(n_epoch, max_epoch, iter, np.mean(train_loss), np.mean(train_err), np.mean(Val_loss), np.mean(Val_err)))

				# if n_incr_error >= patience:
				# 	print ('Early stopping occured. Optimization Finished!')
				# 	return train_loss_history, train_err_history, val_loss_history, val_err_history

			return train_loss_history, train_err_history, val_loss_history, val_err_history

def extract_validation_handles(session):
	""" Extracts the input and predict_op handles that we use for validation.
	Args:
		session: The session with the loaded graph.
	Returns:
		validation handles.
	"""
	valid_nodes = tf.get_collection_ref("validation_nodes")
	if len(valid_nodes) != 5:
		raise Exception("ERROR: Expected 5 items in validation_nodes, got %d." % len(valid_nodes))
	return valid_nodes

def load_model(session, save_path):
	""" Loads a saved TF model from a file.
	Args:
		session: The tf.Session to use.
		save_path: The save path for the saved session, returned by Saver.save().
	Returns:
		The inputs placehoder and the prediction operation.
	"""
	print ("Loading model from file '%s'..." % save_path)

	meta_file = save_path + ".meta"
	if not os.path.exists(meta_file):
		raise Exception("ERROR: Expected .meta file '%s', but could not find it." % meta_file)

	saver = tf.train.import_meta_graph(meta_file)
	# It's finicky about the save path.
	save_path = os.path.join("./", save_path)
	saver.restore(session, save_path)

	# Check that we have the handles we expected.
	return extract_validation_handles(session)

def validate_model(sess, val_names, val_ops, plot_ckpt, batch_size=200):
	""" Validates the model stored in a session.
	Args:
		session: The session where the model is loaded.
		val_data: The validation data to use for evaluating the model.
		val_ops: The validation operations.
	Returns:
		The overall validation error for the model. """

	print ("Validating model...")
	val_num = len(val_names)
	print ("test_num: ", val_num)

	MaxTestIters = int(val_num/batch_size)
	print ("MaxTestIters: ", MaxTestIters)

	val_err = []

	iter_start = None

	eye_left, eye_right, face, face_mask, pred = val_ops
	y = tf.placeholder(tf.float32, [None, 2], name='pos')
	err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred, y), axis=1)))

	for iterTest in range(MaxTestIters):
		test_start=iterTest * batch_size
		test_end = (iterTest+1) * batch_size

		batch_val_data = load_batch_from_data(mtcnn_h, val_names, dataset_path, 1000, img_ch, img_cols, img_rows, train_start = test_start, train_end = test_end)

		batch_val_data = prepare_data(batch_val_data)

		val_batch_err = sess.run(err, feed_dict={eye_left: batch_val_data[0], \
									eye_right: batch_val_data[1], face: batch_val_data[2], \
									face_mask: batch_val_data[3], y: batch_val_data[4]})

		val_err.append(val_batch_err)


		if iterTest % 10 == 0:
			print ('IterTest %s, val error: %.5f' % \
										(iterTest, np.mean(val_err)))

			# plot_loss(np.array(train_loss), np.array(train_err), np.array(Val_err), start=0, per=1, save_file=plot_ckpt + "/testing_loss_" + str(n_epoch) + "_" + str(iterTest) + ".png")

			if iter_start:
				print ('10 iters runtime: %.1fs' % (timeit.default_timer() - iter_start))
			else:
				iter_start = timeit.default_timer()

	return np.mean(val_err)

def plot_loss(train_loss, train_err, test_loss, test_err, start=0, per=1, save_file='loss.png'):
	print ("----plot loss----")

	idx = np.arange(start, len(train_loss), per)
	fig, ax1 = plt.subplots()
	label='train loss'
	lns1 = ax1.plot(idx, train_loss[idx], 'b-', alpha=1.0, label='train loss')
	ax1.set_xlabel('epochs')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('loss', color='b')
	ax1.tick_params('y', colors='b')
	ax1.legend(lns1, label, loc=0)

	fig.tight_layout()
	plt.savefig(save_file + "_train_loss" + ".png")

	fig, ax2 = plt.subplots()
	label='train_err'
	lns2 = ax2.plot(idx, train_err[idx], 'r-', alpha=1.0, label='train_err')
	ax2.set_ylabel('error', color='r')
	ax2.tick_params('y', colors='r')
	ax1.legend(lns2, label, loc=0)

	fig.tight_layout()
	plt.savefig(save_file + "_train_err" + ".png")

	idx = np.arange(start, len(test_loss), per)
	idx_30 = np.arange(start, len(test_loss), per) * 30
	fig, ax1 = plt.subplots()
	label='test loss'
	lns3 = ax1.plot(idx_30, test_loss[idx], 'c-', alpha=1.0, label='test loss')
	ax1.set_xlabel('epochs')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('loss', color='b')
	ax1.tick_params('y', colors='b')
	ax1.legend(lns3, label, loc=0)

	fig.tight_layout()
	plt.savefig(save_file + "_test_loss" + ".png")


	fig, ax2 = plt.subplots()
	label='test_err'
	lns4 = ax2.plot(idx_30, test_err[idx], 'g-', alpha=1.0, label='test_err')
	ax2.set_ylabel('error', color='r')
	ax2.tick_params('y', colors='r')
	ax1.legend(lns4, label, loc=0)

	fig.tight_layout()
	plt.savefig(save_file + "_test_err" + ".png")
	# plt.show()

def train(args):
	start = timeit.default_timer()
	plot_ckpt = "plots/" + date
	if not os.path.exists(plot_ckpt):
		os.makedirs(plot_ckpt)

	out_model = "my_model"
	print ("out_model: ", out_model.split())
	ckpt = out_model + "/" + date + "/"
	print ("ckpt: ", ckpt)
	if not os.path.exists(ckpt):
		os.makedirs(ckpt)


	et = EyeTracker()
	train_loss_history, train_err_history, val_loss_history, val_err_history = et.train(args, ckpt, plot_ckpt, lr=args.learning_rate, batch_size=args.batch_size, max_epoch=args.max_epoch, \
											min_delta=1e-4, \
											patience=args.patience, \
											print_per_epoch=args.print_per_epoch)

	print ('Total training runtime: %.1fs' % (timeit.default_timer() - start))

	plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_err_history), start=0, per=1, save_file= plot_ckpt + "/total_loss.png")

	if args.save_loss:
		with open(plot_ckpt + "/" + args.save_loss, 'w') as outfile:
			np.savez(outfile, train_loss_history=train_loss_history, train_err_history=train_err_history, \
									val_loss_history=val_loss_history, val_err_history=val_err_history)

def test(args):
	print ("--------testing---------")
	plot_ckpt = "plots/" + date
	if not os.path.exists(plot_ckpt):
		os.makedirs(plot_ckpt)

	val_names = load_data_names(val_path)[:2000]
	# Load and validate the network.
	with tf.Session() as sess:
		val_ops = load_model(sess, args.load_model)
		error = validate_model(sess, val_names, val_ops, plot_ckpt, batch_size=args.batch_size)
		print ('Overall validation error: %f' % error)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true', help='train flag')
	# parser.add_argument('-i', '--input', required=True, type=str, help='path to the input data')
	parser.add_argument('-max_epoch', '--max_epoch', type=int, default=60, help='max number of iterations')
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
	# 0.001
	parser.add_argument('-bs', '--batch_size', type=int, default=500, help='batch size')
	parser.add_argument('-p', '--patience', type=int, default=np.Inf, help='early stopping patience')
	parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration')
	parser.add_argument('-sm', '--save_model', type=str, default='my_model', help='path to the output model')
	parser.add_argument('-lm', '--load_model', type=str, help='path to the loaded model')
	parser.add_argument('-pl', '--plot_loss', type=str, default='loss.png', help='plot loss')
	parser.add_argument('-sl', '--save_loss', type=str, default='loss.npz', help='save loss')
	args = parser.parse_args()

	# if args.train:
	train(args)
	# else:
	# 	if not args.load_model:
	# 		raise Exception('load_model arg needed in test phase')
	# test(args)

if __name__ == '__main__':
	main()
