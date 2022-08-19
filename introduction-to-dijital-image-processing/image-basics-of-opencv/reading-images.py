import cv2

image_path = "photos/004.JPG"

# change format to the open-cv photo format

image = cv2.imread(image_path)

cv2.imshow("My Image", image)

# to hold photo until we click the exit button
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
In Open-CV, there is no auto-align.
You'll see the original size

"""