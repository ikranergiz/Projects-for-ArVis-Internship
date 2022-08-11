import cv2

video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    ret, frame = cap.read()

    # frame exits
    if ret:
        cv2.imshow("Frame", frame)

        # Check out this in 25 milliseconds
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # we've ran out ouf frame

    else:
        break
cap.release()
cv2.destroyAllWindows()