import cv2
import detect_face
import tensorflow as tf
import numpy as np

class mtcnn_handle(object):
	def __init__(self):
		sess = tf.Session()
		self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, "../det/")
		self.minsize = 40 # minimum size of face
		self.threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
		self.factor = 0.709 # scale factor


	def run_mtcnn(self, draw, if_face = False, if_facemask = False, if_draw = False):


		#draw = cv2.resize(draw, (960, 540))
		#img=cv2.cvtColor(draw,cv2.COLOR_BGR2GRAY)
		original = draw.copy()
		bounding_boxes, points = detect_face.detect_face(draw, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

		nrof_faces = bounding_boxes.shape[0]

		w, h, _ = draw.shape
		face = []

		for b in bounding_boxes:
			if if_draw:
				cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))

			if if_facemask:
				zeros = np.zeros((w, h))
				face_h, face_w = zeros[int(b[1]):int(b[3]), int(b[0]):int(b[2])].shape
				ones = np.ones((face_h, face_w))
				zeros[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = ones
				face_mask = zeros

			if if_face:
				face = original[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

		if len(points)!=0 and len(face) != 0:
			for p in points.T:
				if if_draw:
					for i in range(5):
						cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

				size = 30
				i = 0

				if if_draw:
					cv2.rectangle(draw, (int(p[i] - size), int(p[i + 5] - size)), (int(p[i] + size), int(p[i + 5] + size)), (0, 255, 0))
				y1, x1 = p[i], p[i + 5]
				left_eye = [p[i], p[i + 5]]
				eye_left = original[int(p[i + 5] - size):int(p[i + 5] + size), int(p[i] - size):int(p[i] + size)]

				i = 1
				if if_draw:
					cv2.rectangle(draw, (int(p[i] - 2 * size), int(p[i + 5] - 2 * size)), (int(p[i] + 2 * size), int(p[i + 5] + 2 * size)), (0, 255, 0))

				x2, y2 = p[i], p[i + 5]
				right_eye = [p[i], p[i + 5]]
				eye_right = original[int(p[i + 5] - size):int(p[i + 5] + size), int(p[i] - size):int(p[i] + size)]

				# if y2 > y1:
				# 	delta_y = y2 - y1
				# 	delta_x = x2 - x1

			# cv2.namedWindow('Face Detection',cv2.WINDOW_NORMAL)
			# cv2.resizeWindow('Face Detection', 1920, 1080)

			if if_draw:
				cv2.imshow('Face Detection',draw)
				cv2.imshow("face", face)
				cv2.imshow("eye_left", eye_left)
				cv2.imshow("eye_right", eye_right)
				cv2.imshow("face_mask", face_mask)
				cv2.imshow("original", original)
				cv2.waitKey(1)

			return [original, draw, face, eye_left, eye_right, face_mask, left_eye, right_eye]
		else:
			return []

def main():
	mtcnn_h = mtcnn_handle()
	cam = cv2.VideoCapture(0)
	while 1:
		ret, frame = cam.read()
		result = mtcnn_h.run_mtcnn(frame,  if_face = True, if_facemask = True, if_draw = True)

if __name__ == '__main__':
	main()
