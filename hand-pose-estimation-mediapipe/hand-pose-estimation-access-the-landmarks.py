import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

image_path = "hand.png"
image = cv2.imread(image_path)

with mp_hands.Hands() as hands:

    # Hand pose estimation
    results = hands.process((cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    # Analysis of the Landmarks
    image_shape = image.shape
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            for point in mp_hands.HandLandmark:
                # Process each landmark
                normalizedLandmark = landmarks.landmark[point]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                       normalizedLandmark.y,
                                                                                       image_shape[0], image_shape[1])
    cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
