import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# to get your cap weight
frame_width = int(cap.get(3))
# to get your cap height
frame_height = int(cap.get(4))

video_cod = cv2.VideoWriter_fourcc(*"XVID")
video_output = cv2.VideoWriter("videos/webcam.avi", video_cod, 30,
                               (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Frame", frame)
        # This function save video by step. It provides us changeable video.
        video_output.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
video_output.release()
cv2.destroyAllWindows()
