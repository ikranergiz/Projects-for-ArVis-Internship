import cv2
import matplotlib.pyplot as plt

image = cv2.imread("photos/004.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# When you crop an image like that, first you need to write height (on the top of left)
# and then weight (on the bottom of left)
cropped_image = image[0:2000, 0:1000]

plt.subplot(121)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(122)
plt.imshow(cropped_image)
plt.title("Cropped Image")

plt.show()