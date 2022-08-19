import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

# 1. Object detection from stable camera
# We add two parameters. To have stable camera, if history raises, you can get more precise detection.
# In my opinion, varThreshold determine the initial limit. So it raises, you'll mostly see more white pixels.
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while cap.isOpened():

    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract region of interest
    roi = frame[340:600, 500:800]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # You need to define "_" variable otherwise error raises like "Can't parse contours".
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    # "-1" means you'd like to draw all contours, so you don't need to use for loop.
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Region of Interest", roi)

    # waitKey(0) hold the image until you press a key.
    # higher than 0, it'll wait dot milliseconds each frame.
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
