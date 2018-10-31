# def live_test(args):
#
# 	global read_window
# 	global mtcnn_window
# 	global data_proc_window
# 	global eye_tracker_window
# 	global display_window
#
#
# 	# cap = cv2.VideoCapture('avi/20180814_15_40_03_357_000_Ch2_Left_F1_750_00000.AVI')
# 	# cap = cv2.VideoCapture('avi/20180814_15_40_03_357_000_Ch3_Right_F1_750_00000.AVI')
# 	# cap = cv2.VideoCapture('demo_data/avi/20180814_15_40_03_357_000_Ch1_Front_F1_750_00000.AVI')
#
# # ------------------------------------------------------------------
#
# 	# cap = cv2.VideoCapture('avi/20180814_15_40_37_002_000_Ch0_Rear_F1_674_00000.AVI')
#
# 	# cap = cv2.VideoCapture('avi/20180814_15_40_37_002_000_Ch1_Front_F1_674_00000.AVI')
# 	# cap = cv2.VideoCapture('avi/20180814_15_40_37_002_000_Ch0_Rear_F1_674_00000.AVI')
# 	# cap = cv2.VideoCapture('demo_data/avi/20180814_15_40_37_002_000_Ch2_Left_F1_674_00000.AVI')
#
# # ------------------------------------------------------------------
#
# 	cap = cv2.VideoCapture('demo_data/avi/20180816_18_24_19_223_000_Ch0_Rear_F1_297_00000.AVI')
#
# # ------------------------------------------------------------------
#
# 	# cap = cv2.VideoCapture('demo_data/avi2/20180829_16_23_41_692_000_F1_593_00000.AVI')
# 	# cap = cv2.VideoCapture('avi2/20180829_16_23_05_405_000_F1_592_00000.AVI')
# 	# cap = cv2.VideoCapture('avi2/20180829_16_22_34_075_000_F1_592_00000.AVI')
# 	# cap = cv2.VideoCapture('demo_data/avi2/20180829_16_21_05_486_000_F1_592_00000.AVI')
#
#
#
#
# 	# cam=cv2.VideoCapture(0)
#
# 	val_ops = load_model(sess, args.load_model)
#
# 	while(cap.isOpened()):
# 	# while 1:
#
# 		start_time = time.time()
#
# 		# path = "demo_data/images/img_Rev22_1280x720_8bit.bmp"
# 		# frame = cv2.imread(path)
# 		ret, frame = cap.read()
# 		# ret, frame = cam.read()
#
# 		lapse = time.time() - start_time
# 		read_window.append(lapse)
# 		if len(read_window) > 10:
# 			read_window = read_window[-10:]
# 		print (" --- read frame ----")
# 		print("--- %s seconds ---" % np.mean(read_window))
# 		start_time = time.time()
#
# 		result = cam_mtcnn(frame)
#
# 		lapse = time.time() - start_time
# 		mtcnn_window.append(lapse)
# 		if len(mtcnn_window) > 10:
# 			mtcnn_window = mtcnn_window[-10:]
# 		print (" --- mtcnn ----")
# 		print("--- %s seconds ---" % np.mean(mtcnn_window))
#
# 		if len(result) > 0:
# 			start_time = time.time()
# 			# print "----------- get result -------------"
# 			[original, draw, face, eye_left, eye_right, face_mask, left_eye, right_eye] = result
#
# 			# print "face_mask.shape"
# 			# print face_mask.shape
#
# 			# disp_img("face_mask_original", face_mask)
# 			# disp_img("face_original", face)
#
# 			face = resize2(face, 64)
# 			eye_left = resize2(eye_left, 64)
# 			eye_right = resize2(eye_right, 64)
# 			face_mask = resize2(face_mask, 25)
#
# 			# disp_img("draw", draw)
# 			# disp_img("face", face)
# 			# disp_img("eye_left", eye_left)
# 			# disp_img("eye_right", eye_right)
# 			# disp_img("face_mask", face_mask)
# 			# disp_img("original", original)
# 			# cv2.waitKey(0)
#
# 			val_data = prepare_data([eye_left, eye_right, face, face_mask, None])
# 			val_data[-1] = np.zeros((1,2))
#
# 			lapse = time.time() - start_time
# 			data_proc_window.append(lapse)
# 			if len(data_proc_window) > 10:
# 				data_proc_window = data_proc_window[-10:]
# 			print (" --- data_proc_window ----")
# 			print("--- %s seconds ---" % np.mean(data_proc_window))
# 			start_time = time.time()
#
# 			# Load and validate the network.
# 			# with tf.Session() as sess:
# 			#     val_ops = load_model(sess, args.load_model)
#
# 			pred = get_prediction(sess, val_data, val_ops)
#
# 			lapse = time.time() - start_time
# 			eye_tracker_window.append(lapse)
# 			if len(eye_tracker_window) > 10:
# 				eye_tracker_window = eye_tracker_window[-10:]
# 			print (" --- eye_tracker ----")
# 			print("--- %s seconds ---" % np.mean(eye_tracker_window))
# 			start_time = time.time()
#
# 			arm_length = 70 # cm
#
# 			px, py = pred[0]
# 			# print (px, py)
# 			increase = 3
# 			# px, py = - int(px * increase), int(py * increase)
# 			px, py = -int(px * increase), int(py * increase)
#
# # TODO: px should times -1 or not???
#
# 			# print (px, py)
#
# 			cv2.line(draw, (int(left_eye[0]), int(left_eye[1])), (int(left_eye[0] + px), int(left_eye[1] + py)), (255, 0, 0), 3)
#
# 			cv2.line(draw, (int(right_eye[0]), int(right_eye[1])), (int(right_eye[0] + px), int(right_eye[1] + py)), (255, 0, 0), 3)
#
#
# 			# draw = resize2(draw, 800)
# 			# cv2.namedWindow('final',cv2.WINDOW_NORMAL)
# 			# cv2.resizeWindow('final', 30, 30)
#
#
# 			cv2.imshow('final', draw)
#
# 			# disp_img("final", draw)
# 			cv2.waitKey(1)
# 			# cv2.waitKey(0)
#
# 			lapse = time.time() - start_time
# 			display_window.append(lapse)
# 			if len(display_window) > 10:
# 				display_window = display_window[-10:]
# 			print (" --- display ----")
# 			print("--- %s seconds ---" % np.mean(display_window))









# def live_test(args):
#
# 	global read_window
# 	global mtcnn_window
# 	global data_proc_window
# 	global eye_tracker_window
# 	global display_window
#
# 	val_ops = load_model(sess, args.load_model)
#
# 	while 1:
#
# 		start_time = time.time()
#
# 		path = "demo_data/images/img_Rev22_1280x720_8bit.bmp"
# 		frame = cv2.imread(path)
#
# 		lapse = time.time() - start_time
# 		read_window.append(lapse)
# 		if len(read_window) > 10:
# 			read_window = read_window[-10:]
# 		print (" --- read frame ----")
# 		print("--- %s seconds ---" % np.mean(read_window))
# 		start_time = time.time()
#
# 		result = cam_mtcnn(frame)
#
# 		lapse = time.time() - start_time
# 		mtcnn_window.append(lapse)
# 		if len(mtcnn_window) > 10:
# 			mtcnn_window = mtcnn_window[-10:]
# 		print (" --- mtcnn ----")
# 		print("--- %s seconds ---" % np.mean(mtcnn_window))
#
# 		if len(result) > 0:
# 			start_time = time.time()
# 			# print "----------- get result -------------"
# 			[original, draw, face, eye_left, eye_right, face_mask, left_eye, right_eye] = result
#
# 			# print "face_mask.shape"
# 			# print face_mask.shape
#
# 			# disp_img("face_mask_original", face_mask)
# 			# disp_img("face_original", face)
#
# 			face = resize2(face, 64)
# 			eye_left = resize2(eye_left, 64)
# 			eye_right = resize2(eye_right, 64)
# 			face_mask = resize2(face_mask, 25)
#
# 			# disp_img("draw", draw)
# 			# disp_img("face", face)
# 			# disp_img("eye_left", eye_left)
# 			# disp_img("eye_right", eye_right)
# 			# disp_img("face_mask", face_mask)
# 			# disp_img("original", original)
# 			# cv2.waitKey(0)
#
# 			val_data = prepare_data([eye_left, eye_right, face, face_mask, None])
# 			val_data[-1] = np.zeros((1,2))
#
# 			lapse = time.time() - start_time
# 			data_proc_window.append(lapse)
# 			if len(data_proc_window) > 10:
# 				data_proc_window = data_proc_window[-10:]
# 			print (" --- data_proc_window ----")
# 			print("--- %s seconds ---" % np.mean(data_proc_window))
# 			start_time = time.time()
#
# 			# Load and validate the network.
# 			# with tf.Session() as sess:
# 			#     val_ops = load_model(sess, args.load_model)
#
# 			pred = get_prediction(sess, val_data, val_ops)
#
# 			lapse = time.time() - start_time
# 			eye_tracker_window.append(lapse)
# 			if len(eye_tracker_window) > 10:
# 				eye_tracker_window = eye_tracker_window[-10:]
# 			print (" --- eye_tracker ----")
# 			print("--- %s seconds ---" % np.mean(eye_tracker_window))
# 			start_time = time.time()
#
# 			arm_length = 70 # cm
#
# 			px, py = pred[0]
# 			# print (px, py)
# 			increase = 3
# 			# px, py = - int(px * increase), int(py * increase)
# 			px, py = -int(px * increase), int(py * increase)
#
# # TODO: px should times -1 or not???
#
# 			# print (px, py)
#
# 			cv2.line(draw, (int(left_eye[0]), int(left_eye[1])), (int(left_eye[0] + px), int(left_eye[1] + py)), (255, 0, 0), 3)
#
# 			cv2.line(draw, (int(right_eye[0]), int(right_eye[1])), (int(right_eye[0] + px), int(right_eye[1] + py)), (255, 0, 0), 3)
#
#
# 			# draw = resize2(draw, 800)
# 			# cv2.namedWindow('final',cv2.WINDOW_NORMAL)
# 			# cv2.resizeWindow('final', 30, 30)
#
#
# 			cv2.imshow('final', draw)
#
# 			# disp_img("final", draw)
# 			cv2.waitKey(0)
#
# 			lapse = time.time() - start_time
# 			display_window.append(lapse)
# 			if len(display_window) > 10:
# 				display_window = display_window[-10:]
# 			print (" --- display ----")
# 			print("--- %s seconds ---" % np.mean(display_window))
