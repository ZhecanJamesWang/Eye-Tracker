import os
import argparse
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import detect_face
from PIL import Image
import time
import os
import datetime
from load_data import load_data_names, load_batch_from_data

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M")

os.environ["CUDA_VISIBLE-DEVICES"] = "1"

read_window = []
mtcnn_window = []
data_proc_window = []
eye_tracker_window = []
display_window = []


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

pre_data_mean = 0


pred_values = None

sess = tf.Session()
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# def disp_img(img, x = None, y = None, px = None, py = None):
#     # img = train_eye_left[0]
#     r, g, b = cv2.split(img)
#     img = cv2.merge([b,g,r])
#     w, h, _= img.shape
#     cx, cy = int(w/2.0), int(h/2.0)
#     cv2.line(img, (cx, cy), (cx + x, cy + y), (0, 0, 255), 3)
#     cv2.line(img, (cx, cy), (cx + px, cy + py), (255, 0, 0), 3)
#
#     cv2.imshow("image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def disp_img(name, img):

	# w, h, _= img.shape
	# cx, cy = int(w/2.0), int(h/2.0)
	# cv2.line(img, (cx, cy), (cx + x, cy + y), (0, 0, 255), 3)

	cv2.imshow(name, img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def resize2(im, desired_size):

	old_size = im.shape[:2] # old_size is in (height, width) format

	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	# new_size should be in (width, height) format

	im = cv2.resize(im, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [0, 0, 0]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
		value=color)

	return new_im

def resize(img, if_facemask = False):

	# You may need to convert the color.
	if not if_facemask:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(img)


	fill_color=(0, 0, 0, 0)

	# print type(img)
	# print im_pil.size

	x, y = im_pil.size
	size = 64
	# new_im = Image.new('RGBA', (size, size), fill_color)
	if not if_facemask:
		new_im = Image.new('RGB', (size, size), fill_color)
	else:
		new_im = Image.new('L', (size, size))

	new_im.paste(im_pil, ((size - x) / 2, (size - y) / 2))

	open_cv_image = np.array(new_im)

	if not if_facemask:
		# Convert RGB to BGR
		result = open_cv_image[:, :, ::-1].copy()
		return result
	else:
		return open_cv_image

def increase_size(img, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized

# Import data
def load_data(file):
	npzfile = np.load(file)

	limit = 100

	train_eye_left = npzfile["train_eye_left"][:limit]
	train_eye_right = npzfile["train_eye_right"][:limit]
	train_face = npzfile["train_face"][:limit]
	train_face_mask = npzfile["train_face_mask"][:limit]
	train_y = npzfile["train_y"][:limit]


	val_eye_left = npzfile["val_eye_left"][:limit]
	val_eye_right = npzfile["val_eye_right"][:limit]
	val_face = npzfile["val_face"][:limit]
	val_face_mask = npzfile["val_face_mask"][:limit]
	val_y = npzfile["val_y"][:limit]

	return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]

def denormalize(data):
	shape = data.shape
	data = np.reshape(data, (shape[0], -1))
	data = data.astype('float32') * 255. # scaling
	data = data + pre_data_mean # normalizing
	return np.reshape(data, shape)

def normalize(data):
	shape = data.shape
	data = np.reshape(data, (shape[0], -1))
	data = data.astype('float32') / 255. # scaling
	pre_data_mean = np.mean(data, axis=0)
	data = data - pre_data_mean # normalizing
	return np.reshape(data, shape)

def prepare_data(data):
	print ("-------- prepare_data --------")
	eye_left, eye_right, face, face_mask, y = data
	eye_left = normalize(eye_left)
	eye_right = normalize(eye_right)
	face = normalize(face)
	face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')

	# if y != None:
	y = y.astype('float32')

	return [eye_left, eye_right, face, face_mask, y]

def shuffle_data(data):
	idx = np.arange(data[0].shape[0])
	np.random.shuffle(idx)
	for i in range(len(data)):
		data[i] = data[i][idx]
	return data

def next_batch(data, batch_size):
	for i in np.arange(0, data[0].shape[0], batch_size):
		# yield a tuple of the current batched data
		yield [each[i: i + batch_size] for each in data]

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
			'fc_face': tf.get_variable('fc_face_w', shape=(face_size, fc_face_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc_face_mask': tf.get_variable('fc_face_mask_w', shape=(mask_size * mask_size, fc_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
			'face_face_mask': tf.get_variable('face_face_mask_w', shape=(fc_face_size + fc_face_mask_size, face_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc': tf.get_variable('fc_w', shape=(fc_eye_size + face_face_mask_size, fc_size), initializer=tf.contrib.layers.xavier_initializer()),
			'fc2': tf.get_variable('fc2_w', shape=(fc_size, fc2_size), initializer=tf.contrib.layers.xavier_initializer())
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
			'fc_face': tf.Variable(tf.constant(0.1, shape=[fc_face_size])),
			'fc_face_mask': tf.Variable(tf.constant(0.1, shape=[fc_face_mask_size])),
			'face_face_mask': tf.Variable(tf.constant(0.1, shape=[face_face_mask_size])),
			'fc': tf.Variable(tf.constant(0.1, shape=[fc_size])),
			'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_size]))
		}

		# Construct model
		self.pred = self.itracker_nets(self.eye_left, self.eye_right, self.face, self.face_mask, self.weights, self.biases)

		tf.summary.histogram("predictions", self.pred)

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

		# face
		face = tf.reshape(face, [-1, int(np.prod(face.get_shape()[1:]))])
		face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc_face']), biases['fc_face']))

		# face mask
		face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc_face_mask']), biases['fc_face_mask']))

		face_face_mask = tf.concat([face, face_mask], 1)
		face_face_mask = tf.nn.relu(tf.add(tf.matmul(face_face_mask, weights['face_face_mask']), biases['face_face_mask']))

		# all
		fc = tf.concat([eye, face_face_mask], 1)
		fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['fc']), biases['fc']))
		out = tf.add(tf.matmul(fc, weights['fc2']), biases['fc2'])
		return out

	def train(self, train_data, val_data, lr=1e-3, batch_size=128, max_epoch=1000, min_delta=1e-4, patience=10, print_per_epoch=10, out_model='my_model', cycle = 0, overall_epoch = 0):


		print ('Train on %s samples, validate on %s samples' % (train_data[0].shape[0], val_data[0].shape[0]))
		# Define loss and optimizer
		self.cost = tf.losses.mean_squared_error(self.y, self.pred)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

		# Evaluate model
		self.err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.pred, self.y), axis=1)))
		train_loss_history = []
		train_err_history = []
		val_loss_history = []
		val_err_history = []
		n_incr_error = 0  # nb. of consecutive increase in error
		best_loss = np.Inf
		n_batches = train_data[0].shape[0] / batch_size + (train_data[0].shape[0] % batch_size != 0)

		# Create the collection
		tf.get_collection("validation_nodes")
		# Add stuff to the collection.
		tf.add_to_collection("validation_nodes", self.eye_left)
		tf.add_to_collection("validation_nodes", self.eye_right)
		tf.add_to_collection("validation_nodes", self.face)
		tf.add_to_collection("validation_nodes", self.face_mask)
		tf.add_to_collection("validation_nodes", self.pred)
		saver = tf.train.Saver(max_to_keep=1)

		# Initializing the variables
		init = tf.global_variables_initializer()
		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			writer = tf.summary.FileWriter("logs", sess.graph)

			# Keep training until reach max iterations
			for n_epoch in range(1, max_epoch + 1):
				n_incr_error += 1
				train_loss = 0.
				val_loss = 0.
				train_err = 0.
				val_err = 0.
				train_data = shuffle_data(train_data)
				for batch_train_data in next_batch(train_data, batch_size):
					# Run optimization op (backprop)
					sess.run(self.optimizer, feed_dict={self.eye_left: batch_train_data[0], \
								self.eye_right: batch_train_data[1], self.face: batch_train_data[2], \
								self.face_mask: batch_train_data[3], self.y: batch_train_data[4]})
					train_batch_loss, train_batch_err = sess.run([self.cost, self.err], feed_dict={self.eye_left: batch_train_data[0], \
								self.eye_right: batch_train_data[1], self.face: batch_train_data[2], \
								self.face_mask: batch_train_data[3], self.y: batch_train_data[4]})
					train_loss += train_batch_loss / n_batches
					train_err += train_batch_err / n_batches
				val_loss, val_err = sess.run([self.cost, self.err], feed_dict={self.eye_left: val_data[0], \
								self.eye_right: val_data[1], self.face: val_data[2], \
								self.face_mask: val_data[3], self.y: val_data[4]})

				train_loss_history.append(train_loss)
				train_err_history.append(train_err)
				val_loss_history.append(val_loss)
				val_err_history.append(val_err)
				if val_loss - min_delta < best_loss:
					print ("out_model: ", out_model.split())

					# ckpt = out_model.split()[0]
					# ckpt = ckpt + "/" + date + "/" + str(cycle) + "/"
					# print ("ckpt: ", ckpt)

					ckpt = os.path.join(os.path.abspath(out_model), date, str(overall_epoch), str(cycle))
					print ("ckpt: ", ckpt)
					if not os.path.exists(ckpt):
						os.makedirs(ckpt)

					ckpt += "/model"
					best_loss = val_loss
					print ("os.path.abspath(out_model): ", os.path.abspath(out_model))

					# , global_step=n_epoch
					save_path = saver.save(sess, ckpt)
					print ("Model saved in file: %s" % save_path)
					n_incr_error = 0

				if n_epoch % print_per_epoch == 0:
					print ('Epoch %s/%s, train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f' % \
												(n_epoch, max_epoch, train_loss, train_err, val_loss, val_err))

				if n_incr_error >= patience:
					print ('Early stopping occured. Optimization Finished!')
					return train_loss_history, train_err_history, val_loss_history, val_err_history

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


# def validate_model(session, val_data, val_ops, orig_val_data = None):
#     """ Validates the model stored in a session.
#     Args:
#         session: The session where the model is loaded.
#         val_data: The validation data to use for evaluating the model.
#         val_ops: The validation operations.
#     Returns:
#         The overall validation error for the model. """
#     print ("Validating model...")
#
#     eye_left, eye_right, face, face_mask, pred = val_ops
#     val_eye_left, val_eye_right, val_face, val_face_mask, val_y = val_data
#     orig_eye_left, orig_eye_right, orig_face, orig_face_mask, orig_y = orig_val_data
#
#     y = tf.placeholder(tf.float32, [None, 2], name='pos')
#     # err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred, y), axis=1)))
#     err = tf.squared_difference(pred, y)
#
#     # err = pred
#     # Validate the model.
#     error = session.run(err, feed_dict={eye_left: val_eye_left, \
#                                 eye_right: val_eye_right, face: val_face, \
#                                 face_mask: val_face_mask, y: val_y})
#
#     pred = session.run(pred, feed_dict={eye_left: val_eye_left, \
#                                 eye_right: val_eye_right, face: val_face, \
#                                 face_mask: val_face_mask, y: val_y})
#     print ("pred.shape")
#     print (pred.shape)
#
#     arm_length = 60 # cm
#
#     images = orig_face
#
#     print (type(pred))
#     # print (type(pred.eval(session)))
#     # tf.Print (pred, [pred])
#     # pred = pred.eval(session)
#     for i in range(len(images)):
#         img = images[i]
#         x, y = orig_y[i]
#         px, py = pred[i]
#         print (px, py)
#         print (x, y)
#         print ("-----------")
#         increase = 2
#         x, y = - int(x * increase), int(y * increase)
#         px, py = - int(px * increase), int(py * increase)
#         print (px, py)
#         print (x, y)
#         img = increase_size(img, 300)
#         disp_img(img, x, y, px, py)
#         # raise "debug"
#
#     raise "debug"
#
#     return error


def get_prediction(session, val_data, val_ops, orig_val_data = None):
	""" Validates the model stored in a session.
	Args:
		session: The session where the model is loaded.
		val_data: The validation data to use for evaluating the model.
		val_ops: The validation operations.
	Returns:
		The overall validation error for the model. """
	print ("Validating model...")


	# print "+++++++++++++++"
	# print val_ops
	# print "---------------"

	# eye_left, eye_right, face, face_mask, pred = val_ops[:5]
	eye_left, eye_right, face, face_mask, pred = val_ops[-5:]

	val_eye_left, val_eye_right, val_face, val_face_mask, val_y = val_data

	val_eye_left = np.expand_dims(val_eye_left, axis=0)
	val_eye_right = np.expand_dims(val_eye_right, axis=0)
	val_face = np.expand_dims(val_face, axis=0)

	val_face_mask = val_face_mask.flatten()
	# print val_face_mask.shape
	val_face_mask = np.expand_dims(val_face_mask, axis=0)
	# print val_face_mask.shape

	y = tf.placeholder(tf.float32, [None, 2], name='pos')

	# err = tf.squared_difference(pred, y)


	# error = session.run(err, feed_dict={eye_left: val_eye_left, \
	#                             eye_right: val_eye_right, face: val_face, \
	#                             face_mask: val_face_mask, y: val_y})

	pred = session.run(pred, feed_dict={eye_left: val_eye_left, \
								eye_right: val_eye_right, face: val_face, \
								face_mask: val_face_mask, y: val_y})

	# print ("pred.shape")
	# print (pred.shape)
	# print (type(pred))



	return pred


def plot_loss(train_loss, train_err, test_err, start=0, per=1, save_file='loss.png'):
	assert len(train_err) == len(test_err)
	idx = np.arange(start, len(train_loss), per)
	fig, ax1 = plt.subplots()
	lns1 = ax1.plot(idx, train_loss[idx], 'b-', alpha=1.0, label='train loss')
	ax1.set_xlabel('epochs')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('loss', color='b')
	ax1.tick_params('y', colors='b')

	ax2 = ax1.twinx()
	lns2 = ax2.plot(idx, train_err[idx], 'r-', alpha=1.0, label='train error')
	lns3 = ax2.plot(idx, test_err[idx], 'g-', alpha=1.0, label='test error')
	ax2.set_ylabel('error', color='r')
	ax2.tick_params('y', colors='r')

	# added these three lines
	lns = lns1 + lns2 + lns3
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc=0)

	fig.tight_layout()
	plt.savefig(save_file)
	# plt.show()

def train(args):
	# train_data, val_data = load_data(args.input)

	# train_size = 10
	# train_data = [each[:train_size] for each in train_data]
	# val_size = 1
	# val_data = [each[:val_size] for each in val_data]


	# dataset_path = "C:\Users\Cheng Lu\Documents\Eye-Tracking-for-Everyone-master\Eye-Tracking-for-Everyone-master\GazeCapture"
	dataset_path = "..\Eye-Tracking-for-Everyone-master\Eye-Tracking-for-Everyone-master\GazeCapture"
	train_path = dataset_path + '\ '.strip() + "train"
	val_path = dataset_path + '\ '.strip() + "validation"
	test_path = dataset_path + '\ '.strip() + "test"

	# train parameters
	# n_epoch = args.max_epoch
	batch_size = 16
	# patience = args.patience

	# image parameter
	img_cols = 64
	img_rows = 64
	img_ch = 3

	# train data
	limit = 1000000000000000
	train_names = load_data_names(train_path)[:limit]
	# validation data
	val_limit = 1000000000000
	val_names = load_data_names(val_path)[:val_limit]
	# test data
	test_names = load_data_names(test_path)[:limit]

	# train_data = prepare_data(train_data)
	# val_data = prepare_data(val_data)
	#
	# print ("train_data: ", type(train_data))
	# print ("train_data: ", len(train_data))
	# print ("train_data: ", train_data.shape)

	start = timeit.default_timer()
	et = EyeTracker()

	train_loss_history = []
	train_err_history = []
	val_loss_history = []
	val_err_history = []
	chunk_size = args.batch_size

	print ("chunk_size: ", chunk_size)

	train_num = len(train_names)
	test_num = len(val_names)

	print ("train_num: ", train_num)
	print ("test_num: ", test_num)

	MaxIters = train_num/chunk_size
	MaxTestIters = test_num/chunk_size

	print ("MaxIters: ", MaxIters)
	print ("MaxTestIters: ", MaxTestIters)

	iterTest=0
	# ////////////////////
	for e in range(args.max_epoch):
		print (" ------------- overall epoch --------------: ", e)
		for iter in range (int(MaxIters)):
			print (" ------------- iter --------------: ", iter)
			train_start=iter * chunk_size
			train_end = (iter+1) * chunk_size

			train_data = load_batch_from_data(train_names, dataset_path, chunk_size, img_ch, img_cols, img_rows, train_start = train_start, train_end = train_end)

			test_start = iterTest * chunk_size
			test_end = (iterTest + 1) * chunk_size

			val_data = load_batch_from_data(val_names, dataset_path, chunk_size, img_ch, img_cols, img_rows, train_start = test_start, train_end = test_end)
			# print (len(batch[0]))
			# print (np.asarray(batch[0][0]).shape)
			# print (batch[1].shape)

			train_loss_history, train_err_history, val_loss_history, val_err_history = et.train(train_data, val_data, \
													lr = args.learning_rate, \
													batch_size = args.batch_size, \
													max_epoch = 1, \
													min_delta = 1e-4, \
													patience = args.patience, \
													print_per_epoch = args.print_per_epoch,
													out_model = args.save_model,\
													cycle = iter, overall_epoch = e)

			train_loss_history.extend(train_loss_history)
			train_err_history.extend(train_err_history)
			val_loss_history.extend(val_loss_history)
			val_err_history.extend(val_err_history)

			iterTest += 1
			iterTest %= MaxTestIters


	tf.summary.histogram("train_loss_history", train_loss_history)
	tf.summary.histogram("train_err_history", train_err_history)
	tf.summary.histogram("val_loss_history", val_loss_history)
	tf.summary.histogram("val_err_history", val_err_history)


	print ('runtime: %.1fs' % (timeit.default_timer() - start))


	# if args.plot_loss:
	plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_err_history), start=0, per=1, save_file=args.plot_loss)

	# if args.save_loss:
	# with open(args.save_loss, 'w') as outfile:
	# 	np.savez(outfile, train_loss_history=train_loss_history, train_err_history=train_err_history, \
	# 							val_loss_history=val_loss_history, val_err_history=val_err_history)

def cam_mtcnn(draw):

	minsize = 40 # minimum size of face
	threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
	factor = 0.709 # scale factor

	#draw = cv2.resize(draw, (960, 540))
	#img=cv2.cvtColor(draw,cv2.COLOR_BGR2GRAY)
	original = draw.copy()
	bounding_boxes, points = detect_face.detect_face(draw, minsize, pnet, rnet, onet, threshold, factor)

	nrof_faces = bounding_boxes.shape[0]

	w, h, _ = draw.shape

	for b in bounding_boxes:
		cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))

		# [y1:y2, x1:x2]
		zeros = np.zeros((w, h))
		face_w = int(b[2]) - int(b[0])
		face_h = int(b[3]) - int(b[1])
		ones = np.ones((face_h, face_w))
		zeros[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = ones
		face_mask = zeros

		face = original[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

	if len(points)!=0:
		for p in points.T:
			for i in range(5):
				cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

		# print (points.shape)
		size = 30
		i = 0
		cv2.rectangle(draw, (int(p[i] - size), int(p[i + 5] - size)), (int(p[i] + size), int(p[i + 5] + size)), (0, 255, 0))

		left_eye = [p[i], p[i + 5]]
		eye_left = original[int(p[i + 5] - size):int(p[i + 5] + size), int(p[i] - size):int(p[i] + size)]

		i = 1
		cv2.rectangle(draw, (int(p[i] - size), int(p[i + 5] - size)), (int(p[i] + size), int(p[i + 5] + size)), (0, 255, 0))

		right_eye = [p[i], p[i + 5]]
		eye_right = original[int(p[i + 5] - size):int(p[i + 5] + size), int(p[i] - size):int(p[i] + size)]


		# cv2.namedWindow('Face Detection',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('Face Detection', 1920, 1080)


		# cv2.imshow('Face Detection',draw)
		# disp_img("face", face)
		# disp_img("eye_left", eye_left)
		# disp_img("eye_right", eye_right)
		# disp_img("face_mask", face_mask)
		# disp_img("original", original)
		#
		# cv2.waitKey(0)
		return [original, draw, face, eye_left, eye_right, face_mask, left_eye, right_eye]
	else:
		return []

def live_test(args):

	global read_window
	global mtcnn_window
	global data_proc_window
	global eye_tracker_window
	global display_window

	cam=cv2.VideoCapture(0)
	# sess = tf.Session()

	val_ops = load_model(sess, args.load_model)

	while 1:

		start_time = time.time()

		ret, frame = cam.read()

		lapse = time.time() - start_time
		read_window.append(lapse)
		if len(read_window) > 10:
			read_window = read_window[-10:]
		print (" --- read frame ----")
		print("--- %s seconds ---" % np.mean(read_window))
		start_time = time.time()

		result = cam_mtcnn(frame)

		lapse = time.time() - start_time
		mtcnn_window.append(lapse)
		if len(mtcnn_window) > 10:
			mtcnn_window = mtcnn_window[-10:]
		print (" --- mtcnn ----")
		print("--- %s seconds ---" % np.mean(mtcnn_window))

		if len(result) > 0:
			start_time = time.time()
			# print "----------- get result -------------"
			[original, draw, face, eye_left, eye_right, face_mask, left_eye, right_eye] = result

			# print "face_mask.shape"
			# print face_mask.shape

			# disp_img("face_mask_original", face_mask)
			# disp_img("face_original", face)

			face = resize2(face, 64)
			eye_left = resize2(eye_left, 64)
			eye_right = resize2(eye_right, 64)
			face_mask = resize2(face_mask, 25)

			# disp_img("draw", draw)
			# disp_img("face", face)
			# disp_img("eye_left", eye_left)
			# disp_img("eye_right", eye_right)
			# disp_img("face_mask", face_mask)
			# disp_img("original", original)
			# cv2.waitKey(0)

			val_data = prepare_data([eye_left, eye_right, face, face_mask, None])
			val_data[-1] = np.zeros((1,2))

			lapse = time.time() - start_time
			data_proc_window.append(lapse)
			if len(data_proc_window) > 10:
				data_proc_window = data_proc_window[-10:]
			print (" --- data_proc_window ----")
			print("--- %s seconds ---" % np.mean(data_proc_window))
			start_time = time.time()

			# Load and validate the network.
			# with tf.Session() as sess:
			#     val_ops = load_model(sess, args.load_model)

			pred = get_prediction(sess, val_data, val_ops)

			lapse = time.time() - start_time
			eye_tracker_window.append(lapse)
			if len(eye_tracker_window) > 10:
				eye_tracker_window = eye_tracker_window[-10:]
			print (" --- eye_tracker ----")
			print("--- %s seconds ---" % np.mean(eye_tracker_window))
			start_time = time.time()

			arm_length = 70 # cm

			px, py = pred[0]
			# print (px, py)
			increase = 3
			px, py = - int(px * increase), int(py * increase)
			# print (px, py)

			cv2.line(draw, (int(left_eye[0]), int(left_eye[1])), (int(left_eye[0] + px), int(left_eye[1] + py)), (255, 0, 0), 3)

			cv2.line(draw, (int(right_eye[0]), int(right_eye[1])), (int(right_eye[0] + px), int(right_eye[1] + py)), (255, 0, 0), 3)


			# draw = resize2(draw, 800)
			# cv2.namedWindow('final',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('final', 30, 30)


			cv2.imshow('final', draw)

			# disp_img("final", draw)
			cv2.waitKey(1)

			lapse = time.time() - start_time
			display_window.append(lapse)
			if len(display_window) > 10:
				display_window = display_window[-10:]
			print (" --- display ----")
			print("--- %s seconds ---" % np.mean(display_window))

def test(args):
	_, val_data = load_data(args.input)
	# val_size = 10
	# val_data = [each[:val_size] for each in val_data]
	orig_val_data = val_data
	val_data = prepare_data(orig_val_data)

	# Load and validate the network.
	with tf.Session() as sess:
		val_ops = load_model(sess, args.load_model)
		error = validate_model(sess, val_data, val_ops, orig_val_data)
		print ('Overall validation error: %f' % error)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true', help='train flag')
	parser.add_argument('-i', '--input', required=False, type=str, help='path to the input data')
	parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations')
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025, help='learning rate')
	parser.add_argument('-bs', '--batch_size', type=int, default = 16, help='batch size')
	parser.add_argument('-p', '--patience', type=int, default=50, help='early stopping patience')
	parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration')
	parser.add_argument('-sm', '--save_model', type=str, default='my_model', help='path to the output model')
	parser.add_argument('-lm', '--load_model', type=str, help='path to the loaded model')
	parser.add_argument('-pl', '--plot_loss', type=str, default='loss.png', help='plot loss')
	parser.add_argument('-sl', '--save_loss', type=str, default='loss.npz', help='save loss')
	args = parser.parse_args()

	if args.train:
		train(args)
	else:
		if not args.load_model:
			raise Exception('load_model arg needed in test phase')
		# test(args)
		live_test(args)

if __name__ == '__main__':
	main()
