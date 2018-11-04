import numpy as np
import random
import cv2
from pylab import array, uint8

def write_to_file(file_name, content):

	fh = open(file_name, "a")
	fh.write(content)
	fh.close

	content = ""
	return content
	
def check_dimension(img, if_even = False, if_last_channel = True):

	if if_last_channel:
		height, width, channels = img.shape
	else:
		height, width = img.shape

	if height == 0:
		raise "left_eye height != width or height == 0 or width == 0"
	if width == 0:
		raise "left_eye height != width or height == 0 or width == 0"

	if if_even:
		if height != width:
			raise "left_eye height != width or height == 0 or width == 0"


def translate(image, w=None, h=None):

	originalImage = image.copy()
	(h, w, _) = image.shape
	xTransRange, yTransRange = np.random.randint(0, w / 6), np.random.randint(0, h / 6)
	newImg = np.zeros_like(image)
	newImg[yTransRange:, xTransRange:] = image[:int(h - yTransRange), :int(w - xTransRange)]
	image = newImg

	return image


def mirror(face, left_eye, right_eye, face_grid, y_x, y_y, h = None, w = None):
    face = np.fliplr(face)
    left_eye = np.fliplr(left_eye)
    right_eye = np.fliplr(right_eye)
    face_grid = np.fliplr(face_grid)
    y_x = w - y_x
    return face, left_eye, right_eye, face_grid, y_x, y_y

def contrastBrightess(image):

    contrast = np.random.uniform(0.5, 3)
    brightness = np.random.uniform(-50, 50)
    # contrast = 2
    # brightness = 50

    maxIntensity = 255.0 # depends on dtype of image data
    phi = 1
    theta = 1
    image = ((maxIntensity/phi)*(image/(maxIntensity/theta))**contrast) + brightness
    image = np.asarray(image)
    top_index = np.where(image > 255)
    bottom_index = np.where(image < 0)
    image[top_index] = 255
    image[bottom_index] = 0
    image = array(image,dtype=uint8)
    return image

def resize(im, desired_size = None):

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

# normalize all data
def normalize(data):

    print("Data normalization...")
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    # scaling
    data = data.astype('float32') / 255.
    # normalizing
    data = data - np.mean(data, axis=0)
    print("Done.")
    return np.reshape(data, shape)


# normalize a single image
def image_normalization(img):

    img = img.astype('float32') / 255.
    # img = img - np.mean(img, axis=0)
    img = img - 127.

    return img


# prepare all data (npz version)
def prepare_data(data):
    print("Data preparing...")
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    print("Done.")
    return [eye_left, eye_right, face, face_mask, y]


# shuffle data
def shuffle_data(data):

    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)
    for i in list(range(len(data))):
        data[i] = data[i][idx]
    return data
