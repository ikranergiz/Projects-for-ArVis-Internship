import cv2

image_path = "photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, None, fx=1 / 5, fy=1 / 5,
                   interpolation=cv2.INTER_AREA)

text = "Hey, There!"
position = (250, 250)
color = (0, 0, 255)
font_size = 1
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(image, text, position, font, font_size, color, thickness)

cv2.imshow("PHOTOS WTIH TEXT", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
