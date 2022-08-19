import cv2
import numpy as np

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)


sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)


cv2.imshow("Canny", edges)
cv2.imshow("Blur Image", img_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()


