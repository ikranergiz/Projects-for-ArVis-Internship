import cv2

image_path = "photos/004.JPG"
image = cv2.imread(image_path)

# BGR to Gray
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Filter", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



"""
Photos we read has BGR format. OpenCV can change the BGR 
format to the RGB format before we see
"""
