import cv2
import mediapipe as mp
import torch
import time
from google.protobuf.json_format import MessageToDict

# Global variables for calculating FPS.
prev_frame_time = 0
new_frame_time = 0

sure = True

# Defining variables for detecting hand pose estimation.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.1) as hands:
    def getColor(x_coordinate, frame_width, y_coordinate, frame_height, webcam_frame):
        """

        :param x_coordinate: bounding box's x coordinate.
        :param frame_width: frame's width.
        :param y_coordinate: bounding box's y coordinate.
        :param frame_height: frame's height.
        :param webcam_frame: frame.
        :return: True means we detect color change signal like mobile-phone screen turns blue (sure = True).
                False means we don't (sure = False).
        """
        blue_lower = (85, 100, 100)
        blue_upper = (135, 255, 255)
        roi = webcam_frame[y_coordinate:frame_height, x_coordinate:frame_width]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, blue_lower, blue_upper)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)

        color_ratio = (mask.sum() // 255) / (roi.shape[0] * roi.shape[1])

        if color_ratio > 0.35:
            return True

        return False


    def calculate_fps(previous_frame_time, current_frame_time, webcam_frame):
        """

        :param previous_frame_time: Previous frame time.
        :param current_frame_time: Current frame time.
        :param webcam_frame: Frame.
        :return: Previous frame time for calculating next FPS value.
        """
        frame_fps = 1 / (current_frame_time - previous_frame_time)
        previous_frame_time = current_frame_time
        frame_fps = int(frame_fps)

        cv2.rectangle(webcam_frame, (10, 20), (200, 60), (200, 50, 10), -1)
        cv2.putText(webcam_frame, f"FPS: {str(frame_fps).upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return previous_frame_time


    def detect_hand_landmarks_and_draw_circle_on_the_fingertip(hand_results_parameter, webcam_frame, frame_width,
                                                               frame_height):
        """

        :param hand_results_parameter: Value of hand process function.
        :param webcam_frame: Frame.
        :param frame_width: Frame's width.
        :param frame_height: Frame's height.
        :return: Fingertip coordinates (x, y).
        """
        # Hand Pose Estimation
        if hand_results_parameter.multi_hand_landmarks:
            for landmark in hand_results_parameter.multi_hand_landmarks:
                mp_drawing.draw_landmarks(webcam_frame, landmark, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                # Process each landmark.
                for point in mp_hands.HandLandmark:
                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        normalized_landmark = landmark.landmark[point]
                        pixel_coordinate = int(normalized_landmark.x * frame_width), int(
                            normalized_landmark.y * frame_height)
                        # Draw red circle on the index finger.
                        cv2.circle(webcam_frame, pixel_coordinate, 20, (0, 0, 255), -2)
                        return pixel_coordinate


    def detect_left_right_hand(hand_results_parameter, webcam_frame):
        """

        :param hand_results_parameter: Value of hand process function.
        :param webcam_frame: Frame.
        """
        cv2.rectangle(webcam_frame, (400, 20), (600, 60), (200, 50, 10), -1)
        # Distinguish left, right or both hand
        if hand_results_parameter.multi_handedness:
            for idx, hand_handedness in enumerate(hand_results_parameter.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                label = handedness_dict["classification"][0]["label"]

                if len(hand_results_parameter.multi_handedness) == 2:
                    cv2.putText(webcam_frame, "Both Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif label == 'Left':
                    cv2.putText(webcam_frame, f"{label} Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2)
                else:
                    cv2.putText(webcam_frame, f"{label} Hands", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2)


    def obtain_bounding_box_coordinates(yolov5_results, webcam_frame):
        """

        :param yolov5_results: Results that returns from YOLOv5.
        :param webcam_frame: Frame.
        :return: Bounding box coordinates (x, y).
        """
        bbox_coordinate = (0, 0)
        df = yolov5_results.pandas().xyxy[0]
        expand_bbox_coordinates = []
        if not df.empty:
            for i in range(len(df)):
                if df.loc[i, "name"] == "Mobile-phone":
                    x1, y1, x2, y2 = df.loc[i, "xmin"], df.loc[i, "ymin"], df.loc[i, "xmax"], df.loc[i, "ymax"]
                    bbox_x = int((x1 + x2) / 2)
                    bbox_y = int((y1 + y2) / 2)
                    bbox_coordinate = (bbox_x, bbox_y)
                    cv2.circle(webcam_frame, bbox_coordinate, 20, (255, 255, 255), -2)
                    expand_bbox_coordinates.append((int(x1), int(x2), int(y1), int(y2)))

        return expand_bbox_coordinates


    # YOLOv5 tiny model loads on local machine.
    model = torch.hub.load(repo_or_dir='yolov5', model='custom', path='yolov5/best.pt',
                           source='local')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty frame!")
            continue

        # FPS
        new_frame_time = time.time()

        # Get frame's width and height.
        width = int(cap.get(3))  # float `width`
        height = int(cap.get(4))  # float `height`

        # ATTENTION! You should convert BGR frame to RGB frame before hand porcess.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #  Get mediapipe hand pose results.
        hand_results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fingertip_coordinate = detect_hand_landmarks_and_draw_circle_on_the_fingertip(hand_results, frame, width,
                                                                                      height)
        detect_left_right_hand(hand_results, frame)

        # Get YOLOv5 results. model function waits BGR format.
        results = model(frame, size=320)

        bbox_coordinate_list = obtain_bounding_box_coordinates(results, frame)
        for bbox_coordinate in bbox_coordinate_list:
            c_x1, c_x2, c_y1, c_y2 = bbox_coordinate
            if getColor(c_x1, c_x2, c_y1, c_y2, frame):
                cv2.putText(frame, "BLUE!", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                if sure:
                    time_function_done = time.time()
                    sure = False

                if time.time() < (time_function_done + 5) and hand_results.multi_hand_landmarks:

                    if (c_x1 < fingertip_coordinate[0] < c_x2) and (c_y1 < fingertip_coordinate[1] < c_y2):
                        cv2.putText(frame, "WELL DONE!", (c_x1, c_y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                                    4)
                        time_function_done = time.time()

                elif time.time() > (time_function_done + 5):
                    cv2.putText(frame, "DO!", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                    if hand_results.multi_hand_landmarks and (c_x1 < fingertip_coordinate[0] < c_x2) and \
                            (c_y1 < fingertip_coordinate[1] < c_y2):
                        time_function_done = time.time()
            else:
                sure = True

        prev_frame_time = calculate_fps(prev_frame_time, new_frame_time, frame)

        results.print()  # print results to screen
        results.save()  # save as results1.jpg, results2.jpg... etc.

        cv2.imshow("Frame", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
