import cv2
import numpy as np

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
image_shape = image.shape

T = np.float32([[1, 0, image_shape[1] / 3],
                [0, 1, image_shape[0] / 3]])

image = cv2.warpAffine(image, T, (image_shape[0], image_shape[1]))


cv2.imshow("Reading Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()