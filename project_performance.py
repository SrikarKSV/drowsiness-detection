import cv2
import numpy as np
from imutils import face_utils
import dlib
import os

# Load custom Haar cascade classifiers
face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_alt.xml")

# Load the CEW dataset
dataset_path = "path/to/CEW_dataset/"
closed_eyes_path = os.path.join(dataset_path, "closed")
open_eyes_path = os.path.join(dataset_path, "open")

# Load the facial landmark predictor
predictor_path = "path/to/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


# Define a function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Initialize some variables for performance measurement
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Loop through the closed eyes folder
for image_name in os.listdir(closed_eyes_path):
    # Load the image and detect faces
    image_path = os.path.join(closed_eyes_path, image_name)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Loop through each face detected
    for x, y, w, h in faces:
        # Crop the face region and detect landmarks
        roi_gray = gray[y : y + h, x : x + w]
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(roi_gray, face_rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Update performance variables based on EAR threshold (here I used 0.25 as the threshold)
        if ear < 0.25:
            true_positives += 1
        else:
            false_negatives += 1

# Loop through the open eyes folder
for image_name in os.listdir(open_eyes_path):
    # Load the image and detect faces
    image_path = os.path.join(open_eyes_path, image_name)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Loop through each face detected
    for x, y, w, h in faces:
        # Crop the face region and detect landmarks
        roi_gray = gray[y : y + h, x : x + w]
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(roi_gray, face_rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Update performance variables based on EAR threshold (here I used 0.25 as the threshold)
        if ear >= 0.25:
            true_negatives += 1
        else:
            false_positives += 1

total_images = true_positives + false_positives + true_negatives + false_negatives
print("Total images:", total_images)
print("True positives:", true_positives)
print("False positives:", false_positives)
print("True negatives:", true_negatives)
print("False negatives:", false_negatives)
print("Accuracy:", (true_positives + true_negatives) / total_images)
print("Precision:", true_positives / (true_positives + false_positives))
print("Recall:", true_positives / (true_positives + false_negatives))
print(
    "F1 Score:",
    2 * (true_positives / (2 * true_positives + false_positives + false_negatives)),
)
