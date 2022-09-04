import cv2
import mediapipe as mp
import torch
import time
from google.protobuf.json_format import MessageToDict


def getColor(x, w, y, h, frame):
    blueLower = (85, 100, 100)
    blueUpper = (135, 255, 255)
    kare = frame[y:y + h, x:x + w]
    # cv2.imwrite("box.jpg", kare)
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

    maske = cv2.inRange(hsv, blueLower, blueUpper)
    cv2.imwrite("maske1.jpg", maske)
    maske = cv2.dilate(maske, None, iterations=2)
    cv2.imwrite("maske_dilated.jpg", maske)
    maske = cv2.erode(maske, None, iterations=2)
    cv2.imwrite("maske2.jpg", maske)

    color_ratio = (maske.sum() // 255) / (kare.shape[0] * kare.shape[1])
    # print(maske.sum() // 255, kare.shape[0] * kare.shape[1])

    if color_ratio > 0.35:
        return True
    else:
        return False


prev_frame_time = 0
new_frame_time = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

x = None
y = None
sure = True
# Model
model = torch.hub.load(repo_or_dir='yolov5', model='custom', path='object-detection-roboflow-train-yolov5/best.pt',
                       source='local')  # local repo

with mp_hands.Hands(
        # If set to false, the solution treats the input images as a video stream.
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.1
        # min_tracking_confidence=0.1
) as hands:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            continue

        # FPS
        new_frame_time = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame, size=320)

        width = int(cap.get(3))  # float `width`
        height = int(cap.get(4))  # float `height`

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Left or Right
        if hand_results.multi_handedness:
            for idx, hand_handedness in enumerate(hand_results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                # print(handedness_dict["classification"][0]["label"])
                right_hand = handedness_dict["classification"][0]["label"]

        # Hand Pose Estimation
        if hand_results.multi_hand_landmarks:
            for landmark in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Process each landmark
                for point in mp_hands.HandLandmark:
                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        normalizedLandmark = landmark.landmark[point]
                        coordinate = int(normalizedLandmark.x * width), int(normalizedLandmark.y * height)
                        print("FINGERTIP: ", (coordinate))
                        frame = cv2.circle(frame, coordinate, 20, (255, 0, 0), -3)

        # Distinguish left, right or both hand
        if hand_results.multi_handedness:
            for idx, hand_handedness in enumerate(hand_results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                # print(handedness_dict["classification"][0]["label"])
                label = handedness_dict["classification"][0]["label"]
                if len(hand_results.multi_handedness) == 2:
                    cv2.putText(frame, "Both Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif label == 'Left':
                    cv2.putText(frame, f"{label} Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, f"{label} Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        df = results.pandas().xyxy[0]
        if not df.empty:
            for i in range(len(df)):
                if df.loc[i, "name"] == "Mobile-phone":
                    x1, y1, x2, y2 = df.loc[i, "xmin"], df.loc[i, "ymin"], df.loc[i, "xmax"], df.loc[i, "ymax"]
                    print(x1, y1, x2, y2)
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)
                    frame = cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)
        if x is not None:
            if getColor(x, width, y, height, frame):
                if sure:
                    print("calistim")
                    cv2.putText(frame, "WORK!", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    time_function_done = time.time()
                    sure = False

                if time.time() < (time_function_done + 5):
                        time_function_done = time.time()
                else:
                    print(time_function_done, time.time())
                    cv2.putText(frame, "DO!", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    sure = True
            else:
                # print("maviler patladi")
                cv2.putText(frame, "BLUE!", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                sure = True

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)

        # You should put below code if you want to see bounding boxes.
        # Results
        results.print()  # print results to screen
        # results.show()  # display results
        results.save()  # save as results1.jpg, results2.jpg... etc.
        
        cv2.rectangle(frame, (10, 10), (200, 50), (200, 50, 10), -1)
        cv2.putText(frame, f"FPS: {str(fps).upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
