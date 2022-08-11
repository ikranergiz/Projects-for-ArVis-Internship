import cv2

image_path = "photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, None, fx=1 / 5, fy=1 / 5,
                   interpolation=cv2.INTER_AREA)

image_shape = image.shape

# it must be int format
center_point = (int(image_shape[0] * 0.5), int(image_shape[1] * 0.5))
radius = 100

cv2.circle(image, center_point, radius, (255, 0, 0), thickness = 5)


cv2.imshow("PHOTOS WTIH TEXT", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
