import cv2
import numpy as np

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
image_shape = image.shape

image1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
image2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
image3 = cv2.rotate(image, cv2.ROTATE_180)

cv2.imshow("Image1", image1)
cv2.imshow("Image2", image2)
cv2.imshow("Image3", image3)

# to concatenate images, it should have same dimension
image_concat = np.concatenate((image1,image2, image3), axis=1)
cv2.imshow("Image Concat", image_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()