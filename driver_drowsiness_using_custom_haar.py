# Importing OpenCV Library for basic image processing functions
import cv2

# Numpy for array related functions
import numpy as np

# Dlib for deep learning based Modules and face landmark detection
import dlib

# face_utils for basic operations of conversion
from imutils import face_utils

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Loading the face classifier
face_classifier = cv2.CascadeClassifier("face_classifier.xml")

# Initializing the landmark detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces using the classifier
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    face_frame = frame.copy()
    # detected face in faces array
    for x, y, w, h in faces:
        cv2.rectangle(face_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the face region to grayscale for landmark detection
        face_gray = gray[y : y + h, x : x + w]

        landmarks = predictor(face_gray, dlib.rectangle(0, 0, w, h))
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(
            landmarks[36],
            landmarks[37],
            landmarks[38],
            landmarks[41],
            landmarks[40],
            landmarks[39],
        )
        right_blink = blinked(
            landmarks[42],
            landmarks[43],
            landmarks[44],
            landmarks[47],
            landmarks[46],
            landmarks[45],
        )

        # Now judge what to do for the eye blinks
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (color),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Left eye Blinking : {}".format(left_blink),
            (200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Right eye Blinking : {}".format(right_blink),
            (200, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Show the output frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
