import cv2

cap = cv2.VideoCapture("highway.mp4")

# Object detection from stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():

    ret, frame = cap.read()

    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # It works because you put the output of function into 2 different variable called "contours" and "hierarchy"
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
