import cv2
import sys

(major_ver, minor_ver, subminor_ver) =  (cv2.__version__).split('.')

# __name__ is a built-in variable that evaluates the name of the current module.
if __name__ == '__main__':

    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[0]

    # Check the version.
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)

    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()

    # to see version of installed opencv.
    print(major_ver, minor_ver,subminor_ver)


# cap = cv2.VideoCapture("../object-tracking-with-stable-camera/highway.mp4")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open video!")
    sys.exit()

ret, frame = cap.read()
if not ret:
    print("Cannot read video file!")
    sys.exit()

# creating region of interest (ROI)
# bbox = cv2.selectROI(frame, False)
bbox = (100, 30, 60, 30)

# Initialize tracker with first frame and bounding box.
ok = tracker.init(frame, bbox)

while cap.isOpened():

    ret, frame = cap.read()

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ret, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ret:
        # Draw bounding box.
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2, 1)

        cv2.putText(frame, tracker_type + "Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, "FPS" + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        # Check out this in 25 milliseconds
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        break

cap.release()
cv2.destroyAllWindows()