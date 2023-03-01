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
with open("bg.txt", "a") as f:
    for img in os.listdir(neg_dir):
        line = os.path.join(neg_dir, img) + "\n"
        f.write(line)
        num_of_neg_samples += 1

num_of_pos_samples = 0
with open("pos.info", "a") as f:
    for img in os.listdir(pos_dir):
        line = os.path.join(pos_dir, img) + f" 1 0 0 {width} {height}\n"
        f.write(line)
        num_of_pos_samples += 1

# Create the positive samples vector file
img_filenames = [line.split()[0] for line in open("pos.info").readlines()]
with open("pos.vec", "wb") as f:
    for filename in img_filenames:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (width, height))
        cv2.imwrite("temp.png", resized_img)
        with open("temp.png", "rb") as temp:
            data = temp.read()
            f.write(np.array([len(data)], dtype=np.int32).tobytes() + data)
    os.remove("temp.png")


# Train the Haar Cascade classifier
params = {
    "numPos": num_of_pos_samples,
    "numNeg": 0,  # no negative images used in opencv_createsamples command
    "numStages": 20,
    "minHitRate": 0.999,
    "maxFalseAlarmRate": 0.5,
    "w": 25,
    "h": 25,
    "maxxangle": 0.7,
    "maxyangle": 0.7,
    "maxzangle": 0.7,
    "vec": "pos.vec",
}

classifier = cv2.CascadeClassifier()
classifier.train("bg.txt", params=params)

# Save the trained classifier
classifier.save("face_classifier.xml")
