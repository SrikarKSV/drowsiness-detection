import cv2
import os
import numpy as np

# Define paths to positive and negative image directories
pos_dir = "path/to/positive/images"
neg_dir = "path/to/negative/images"

# Define dimensions of images
width = 25
height = 25

num_of_neg_samples = 0
for img in os.listdir(neg_dir):
    line = neg_dir + img + "\n"
    with open("bg.txt", "a") as f:
        f.write(line)

    num_of_neg_samples += 1

num_of_pos_samples = 0
for img in os.listdir(pos_dir):
    line = pos_dir + img + " 1 0 0 96 96\n"

    with open("pos.info", "a") as f:
        f.write(line)

    num_of_pos_samples += 1


# Create the positive samples vector file
os.system(
    f"opencv_createsamples -info pos.info -num {num_of_pos_samples} -w {width} -h {height} -maxxangle 0.7 -maxyangle 0.7 -maxzangle 0.7 -vec pos.vec"
)

# Train the Haar Cascade classifier
params = {
    "numPos": num_of_pos_samples,
    "numNeg": num_of_neg_samples,
    "numStages": 20,
    "featureType": cv2.HaarCascadeClassifier.LBP,
    "minHitRate": 0.999,
    "maxFalseAlarmRate": 0.5,
    "w": width,
    "h": height,
    "numThreads": 2,
    "precalcValBufSize": 5120,
    "precalcIdxBufSize": 5120,
}

classifier = cv2.CascadeClassifier()
classifier.train("pos.vec", "bg.txt", params=params)

# Save the trained classifier
classifier.save("face_classifier.xml")
