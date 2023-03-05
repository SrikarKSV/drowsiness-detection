import cv2
import numpy as np
import os

# Load the Haar Cascade file and create the classifier
cascade_path = "path/to/haar/cascade.xml"
cascade_classifier = cv2.CascadeClassifier(cascade_path)

# Define the directory containing positive and negative test images
positive_dir = "pos/"
negative_dir = "neg/"

# Define the true/false positive/negative counters
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Loop over the positive test images and detect objects using the Haar Cascade
for filename in os.listdir(positive_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(positive_dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            true_positives += 1
        else:
            false_negatives += 1

# Loop over the negative test images and detect objects using the Haar Cascade
for filename in os.listdir(negative_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(negative_dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            false_positives += 1
        else:
            true_negatives += 1

# Calculate the statistics
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * ((precision * recall) / (precision + recall))

# Print the results
print("True Positives: {}".format(true_positives))
print("False Positives: {}".format(false_positives))
print("True Negatives: {}".format(true_negatives))
print("False Negatives: {}".format(false_negatives))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1_score))
