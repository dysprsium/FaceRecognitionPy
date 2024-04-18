import pathlib
import cv2
import time

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

face_detected_time = None
delay_time = 3  # Delay time in seconds

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
        if face_detected_time is None:
            face_detected_time = time.time()
        elif time.time() - face_detected_time >= delay_time:
            cv2.putText(frame, "Granted", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
