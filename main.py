import argparse
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# To convert to grayscale with cv2
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# cv2.waitKey(0)

# To convert to grayscale with plt

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(121)
plt.imshow(image)
plt.title("Tetris Blocks")

plt.subplot(122)
plt.imshow(gray_image, cmap='gray')
plt.title("Gray Tetris Block")
plt.show()


# NOTE: to convert your image to grayscale
# If you use cv2.cvtColor to convert to grayscale, you should use cv2.imshow.
# In spite of using cv2.imshow, you might use plt.imshow, but cv2.cvtColor won't work so
# you need to write this -> plt.imshow(image, cmap='gray'). Be careful.
