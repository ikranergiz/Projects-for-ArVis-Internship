import cv2

# OpenCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# Creating classes array to write as text labels
with open('dnn_model/classes.txt') as f:
    lines = f.readlines()

classes = []
for line in lines:
    classes.append(line.strip("\n").capitalize())

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        continue

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    # Using zip function, we don't need to use 3 nested for loops.
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        start_point = (int(x), int(y))
        end_point = (int(x + w), int(y + h))

        class_name = classes[class_id]
        # Drawing bounding boxes
        cv2.rectangle(frame, start_point, end_point, (200, 0, 100), 3)
        # Drawing background
        cv2.rectangle(frame, (x + 175, y - 45), (x - 1, y - 1), (200, 0, 100), -1)
        # cv2.rectangle(image=frame, start_point=start_point, end_point=end_point, color=(0, 0, 255), thickness=2)

        # Writing class name
        cv2.putText(frame, class_name, (x, y - 10), 2, cv2.FONT_HERSHEY_PLAIN, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
