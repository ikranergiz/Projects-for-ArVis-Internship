import cv2

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, None, fx=1/5, fy=1/5,
           interpolation=cv2.INTER_AREA)

x, y, w, h = 300, 500, 700, 750

cropped_image = image[y:y+h, x:x+w]


cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

