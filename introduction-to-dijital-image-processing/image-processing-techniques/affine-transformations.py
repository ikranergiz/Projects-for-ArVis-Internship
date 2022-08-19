import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
image_shape = image.shape

rows, cols, ch = image.shape

plts1 = np.float32([[50, 50],
                    [20, 50],
                    [50, 200]
                    ])

plts2 = np.float32([[10, 10],
                    [200, 50],
                    [100, 250]
                    ])

M = cv2.getAffineTransform(plts1, plts2)
dst = cv2.warpAffine(image, M, (cols, rows))

plt.subplot(121)
plt.imshow(image)
plt.title("Input")

plt.subplot(122)
plt.imshow(image)
plt.title("Output")

plt.show()