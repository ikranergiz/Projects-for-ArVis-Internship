import cv2
import numpy as np

image_path = "../photos/004.JPG"
image = cv2.imread(image_path)
image_shape = image.shape

plts1 = np.float32([[0, 250],
                    [620, 250],
                    [0, 400],
                    [610, 340]
                    ])

plts2 = np.float32([[0, 0],
                    [400, 0],
                    [0,640],
                    [400, 634]
                    ])

M = cv2.getPerspectiveTransform(plts1, plts2)
result = cv2.warpPerspective(image, M, (500, 600))

cv2.imshow("Input", image)
cv2.imshow("Output", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
