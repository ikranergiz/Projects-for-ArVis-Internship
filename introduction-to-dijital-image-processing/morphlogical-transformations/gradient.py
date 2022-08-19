import cv2
import numpy as np

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (512, 512))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

blue1 = np.array([100, 0, 0])
blue2 = np.array([255, 255, 255])

mask = cv2.inRange(hsv, blue1, blue2)

res = cv2.bitwise_and(image, image, mask=mask)
kernel = np.ones((5, 5), np.uint8)

gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

cv2.imshow("Image", image)
cv2.imshow("Final Image", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()