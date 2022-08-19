import cv2

image_path = "photos/004.JPG"
image = cv2.imread(image_path)

new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imwrite("photos/gray_004.JPG", new_image)
gray_image_path = "photos/gray_004.JPG"
new_gray_image = cv2.imread(gray_image_path)

cv2.imshow("Gray Image", new_gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
