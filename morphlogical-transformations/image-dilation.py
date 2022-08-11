import cv2
import numpy as np

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))

kernel = np.ones((10, 10), np.uint8)

dilated_image = cv2.dilate(image, kernel, iterations=1)

concat_image = np.concatenate((image, dilated_image), axis=1)

cv2.imshow("Final image", concat_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


