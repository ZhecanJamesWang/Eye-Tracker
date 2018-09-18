from utility.data_utility import image_normalization, resize, check_dimension
import cv2

img = cv2.imread("test.png")

img = resize(img, 400)
cv2.imshow("first", img)
img = resize(img, 100)
cv2.imshow("second", img)

cv2.waitKey(0)
