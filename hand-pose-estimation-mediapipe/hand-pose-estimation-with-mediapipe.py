SCREEN_DIMENSIONS = (1920, 1080)


# Source: https://stackoverflow.com/questions/68686155/how-to-convert-normalized-coordinates-to-pixel-coordinates
def to_pixel_coords(relative_coords):
    return tuple(
        round(coord * dimension) for coord, dimension in zip(relative_coords, SCREEN_DIMENSIONS))


import time
import cv2
import mediapipe as mp

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

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # It is important to take in consideration that the process method receives a RGB image but,
        # when reading images with OpenCV, we obtain them in BGR format.

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
                        coordinate = (normalizedLandmark.x, normalizedLandmark.y)
                        c1 = to_pixel_coords(coordinate)
                        print("FINGERTIP: ", c1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
