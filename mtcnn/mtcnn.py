import cv2
from mtcnn import detect_face
import tensorflow as tf
import numpy as np

class mtcnn_handle(object):
	def __init__(self):
		sess = tf.Session()
		self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, "det/")
		# self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, "../det/")

		self.minsize = 40 # minimum size of face
		self.threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
		self.factor = 0.709 # scale factor


	def run_mtcnn(self, draw, if_face = False, if_facemask = False, if_draw = False):


		original = draw.copy()
		bounding_boxes, points = detect_face.detect_face(draw, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

		w, h, _ = draw.shape
		face, face_mask = None, None

		if if_draw:
			b = bounding_boxes[0]
			cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))

		if if_facemask:
			b = bounding_boxes[0]
			zeros = np.zeros((w, h))
			face_h, face_w = zeros[int(b[1]):int(b[3]), int(b[0]):int(b[2])].shape
			ones = np.ones((face_h, face_w))
			zeros[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = ones
			face_mask = zeros

		if if_face:
			face = original[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

		# print "points.T: ", points.T
		p = points.T[0]

		if if_draw:
			for i in range(5):
				cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

		i = 0
		x1, y1 = p[i], p[i + 5]
		i = 1
		x2, y2 = p[i], p[i + 5]

		delta_y = abs(y2 - y1)
		delta_x = abs(x2 - x1)
		theta = np.arctan2(delta_y, delta_x)
		l = np.sqrt(delta_x**2 + delta_y**2)
		l_prime = (7.0/15.0)*(1.0/2.0)*l

		x_prime = np.cos(theta) * l_prime
		padding = int(x_prime * 1.5)

		left_eye_pts = [x1, y1]
		left_eye = original[int(y1 - padding):int(y1 + padding), int(x1 - padding):int(x1 + padding)]

		right_eye_pts = [x2, y2]
		right_eye = original[int(y2 - padding):int(y2 + padding), int(x2 - padding):int(x2 + padding)]

		if if_draw:
			cv2.rectangle(draw, (int(x1 - padding), int(y1 - padding)), (int(x1 + padding), int(y1 + padding)), (0, 255, 0))
			cv2.rectangle(draw, (int(x2 - padding), int(y2 - padding)), (int(x2 + padding), int(y2 + padding)), (0, 255, 0))

			cv2.imshow('Face Detection',draw)
			# cv2.namedWindow('Face Detection',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('Face Detection', 1920, 1080)
			cv2.imshow("face", face)
			cv2.imshow("eye_left", eye_left)
			cv2.imshow("eye_right", eye_right)
			cv2.imshow("face_mask", face_mask)
			cv2.imshow("original", original)
			cv2.waitKey(0)

		return [original, draw, face, left_eye, right_eye, face_mask, left_eye_pts, right_eye_pts]

def main():
	mtcnn_h = mtcnn_handle()
	cam = cv2.VideoCapture(0)
	while 1:
		ret, frame = cam.read()
		result = mtcnn_h.run_mtcnn(frame,  if_face = True, if_facemask = True, if_draw = True)

	# ------------------------
	# mtcnn_h = mtcnn_handle()
	# img = cv2.imread("02426_00504.jpg_image.png")
	# result = mtcnn_h.run_mtcnn(img,  if_face = True, if_facemask = True, if_draw = True)
	# result = mtcnn_h.run_mtcnn(img,  if_face = False, if_facemask = False, if_draw = False)
	# [_, draw, _, left_eye, right_eye, _, left_eye_pts, right_eye_pts] = result
	# cv2.imshow('Face Detection',draw)
	# cv2.waitKey(0)

if __name__ == '__main__':
	main()
