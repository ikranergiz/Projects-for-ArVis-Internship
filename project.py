import cv2
import mediapipe as mp
import torch
import time

prev_frame_time = 0
new_frame_time = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Model
model = torch.hub.load(repo_or_dir='yolov5', model='custom', path='object-detection-roboflow-train-yolov5/best.pt', source='local')  # local repo

with mp_hands.Hands(
        # If set to false, the solution treats the input images as a video stream.
        static_image_mode=False,
        max_num_hands=2,
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

        width = int(cap.get(3))  # float `width`
        height = int(cap.get(4))  # float `height`

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.rectangle(frame, (10, 10), (200, 50), (200, 50, 10), -1)
        cv2.putText(frame, f"FPS: {str(fps).upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Hand Pose Estimation
        if results.multi_hand_landmarks:

            for landmark in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Process each landmark
                for point in mp_hands.HandLandmark:
                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        normalizedLandmark = landmark.landmark[point]
                        coordinate = int(normalizedLandmark.x * width), int(normalizedLandmark.y * height)
                        print("FINGERTIP: ", (coordinate))
                        frame = cv2.circle(frame, coordinate, 20, (0, 0, 255), -3)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)

        cv2.rectangle(frame, (10, 10), (200, 50), (200, 50, 10), -1)
        cv2.putText(frame, f"FPS: {str(fps).upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame, size=320)
        # Results
        results.print()

        results.save()  # or .show(

        cv2.imshow("Frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
