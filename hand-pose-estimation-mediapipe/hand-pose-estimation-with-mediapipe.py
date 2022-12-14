import cv2
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict

prev_frame_time = 0
new_frame_time = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        # If set to false, the solution treats the input images as a video stream.
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.1
        # min_tracking_confidence=0.1
) as hands:
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Ignoring the empty frame!")
            continue

        # FPS
        new_frame_time = time.time()
        width = int(cap.get(3))  # float `width`
        height = int(cap.get(4))  # float `height`

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # It is important to take in consideration that the process method receives a RGB image but,
        # when reading images with OpenCV, we obtain them in BGR format.

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        # print('Handedness:', results.multi_handedness)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Left or Right
        if results.multi_handedness is not None:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                # print(handedness_dict["classification"][0]["label"])
                right_hand = handedness_dict["classification"][0]["label"]

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

        # Distinguish left, right or both hand
        if results.multi_handedness:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                # print(handedness_dict["classification"][0]["label"])
                label = handedness_dict["classification"][0]["label"]
                if len(results.multi_handedness) == 2:
                    cv2.putText(frame, "Both Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif label == 'Left':
                    cv2.putText(frame, f"{label} Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, f"{label} Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)

        cv2.rectangle(frame, (10, 10), (200, 50), (200, 50, 10), -1)
        cv2.putText(frame, f"FPS: {str(fps).upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
