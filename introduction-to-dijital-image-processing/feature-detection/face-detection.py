import cv2

face_cascade = cv2.CascadeClassifier("../haarcascade/haarcascade_frontalcatface.xml")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if ret:

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        faces = face_cascade.detectMultiScale(img_gray)

        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            image = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), thickness=4)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
