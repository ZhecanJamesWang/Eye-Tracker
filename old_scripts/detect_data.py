def load_batch_from_data(names, path, batch_size, img_ch, img_cols, img_rows, train_start = None, train_end = None):

	save_img = False

	# data structures for batches
	left_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	right_eye_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	face_batch = np.zeros(shape=(batch_size, img_cols, img_rows, img_ch), dtype=np.float32)
	face_grid_batch = np.zeros(shape=(batch_size, 25, 25), dtype=np.float32)
	y_batch = np.zeros((batch_size, 2), dtype=np.float32)

	# counter for check the size of loading batch
	b = 0
	# while b < batch_size:
	# print ("int(train_start),int(train_end: ", int(train_start),int(train_end))

	for i in range(int(train_start),int(train_end)):
		try:
			# lottery
			# i = np.random.randint(0, len(names))

			# get the lucky one
			img_name = names[i]
		except Exception as e:
			print (e)

		# directory
		dir = img_name[:5]

		# frame name
		frame = img_name[6:]

		# index of the frame into a sequence
		idx = int(frame[:-4])
		# print ("img_name: ", img_name)
		# print ("frame: ", frame)
		# print ("idx: ", idx)
		# print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
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
		# face_grid = np.zeros(shape=(25, 25, 1), dtype=np.float32)
		face_grid = np.zeros(shape=(25, 25), dtype=np.float32)
		tl_x = int(grid_json["X"][idx])
		tl_y = int(grid_json["Y"][idx])
		w = int(grid_json["W"][idx])
		h = int(grid_json["H"][idx])
		br_x = tl_x + w
		br_y = tl_y + h

		# print ("face_grid: ", face_grid.shape)
		# face_grid[0, tl_y:br_y, tl_x:br_x] = 1
		face_grid[tl_y:br_y, tl_x:br_x] = 1
		# face_grid = cv2.resize(face_grid,(25, 25))
		# face_grid = np.array(face_grid * 255, dtype = np.uint8)
		# face_grid = face_grid.flatten()

		# get labels
		y_x = dot_json["XCam"][idx]
		y_y = dot_json["YCam"][idx]


		# cv2.imwrite("images/" + dir + "_" + frame + "_face.png", face)
		# cv2.imwrite("images/" + dir + "_" + frame + "_right.png", right_eye)
		# cv2.imwrite("images/" + dir + "_" + frame + "_left.png", left_eye)
		# cv2.imwrite("images/" + dir + "_" + frame + "_faceGrid.png", face_grid)
		# cv2.imwrite("images/" + dir + "_" + frame + "_image.png", img)
		#
		# print ("face.shape: ", face.shape)
		# print ("left_eye.shape: ", left_eye.shape)
		# print ("right_eye.shape: ", right_eye.shape)


# ///////////////////////////////////////////////////
		# resize images
		# h, w, _ = face.shape
		# print ("vvvvvvvvvvvvvvvvvvv")
		# print ("face.shape: ", face.shape)

		face = cv2.resize(face, (img_cols, img_rows))
		left_eye = cv2.resize(left_eye, (img_cols, img_rows))
		right_eye = cv2.resize(right_eye, (img_cols, img_rows))

		# scale = img_cols/w
		#
		# print (y_x, y_y)
		# print ("scale: ", scale)
		# y_x *= scale
		# y_y *= scale
		# print (y_x, y_y)
		# print ("/\/\/\/\/\/\/\//\/\/")



#
# 		# save images (for debug)
# # /////////////////////////////////////////////////////////
# 		# if save_img:
		# increase = 3
		# y_x, y_y = - int(y_x * increase), int(y_y * increase)
		# # print (px, py)
		# h, w, _ = face.shape
		# cx, cy = w/2.0, h/2.0
		# cv2.circle(face,(int(cx), int(cy)), 5, (0,0,255), -1)
		# cv2.line(face, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)
		#

		# normalization
		# face = image_normalization(face)
		# left_eye = image_normalization(left_eye)
		# right_eye = image_normalization(right_eye)

		######################################################


		# transpose images
		# face = face.transpose(2, 0, 1)
		# left_eye = left_eye.transpose(2, 0, 1)
		# right_eye = right_eye.transpose(2, 0, 1)


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

	# return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch
	return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch, y_batch]
