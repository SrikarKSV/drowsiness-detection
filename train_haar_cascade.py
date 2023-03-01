import cv2
import os
import numpy as np

# Define paths to positive and negative image directories
pos_dir = "path/to/positive/images"
neg_dir = "path/to/negative/images"

# Define dimensions of positive images
width = 24
height = 24


# Read in the images and convert to grayscale
def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# Extract features from positive images
def extract_features(pos_dir, width, height):
    features = []
    for filename in os.listdir(pos_dir):
        img = read_image(os.path.join(pos_dir, filename))
        resized_img = cv2.resize(img, (width, height))
        features.append(resized_img)
    return features


# Extract features from negative images
def extract_negative_features(neg_dir, width, height):
    features = []
    for filename in os.listdir(neg_dir):
        img = read_image(os.path.join(neg_dir, filename))
        resized_img = cv2.resize(img, (width, height))
        features.append(resized_img)
    return features


# Create positive samples
def create_samples(features):
    samples = []
    for img in features:
        x = np.random.randint(0, img.shape[1] - width)
        y = np.random.randint(0, img.shape[0] - height)
        sample = img[y : y + height, x : x + width]
        samples.append(sample)
    return samples


# Extract positive and negative features
pos_features = extract_features(pos_dir, width, height)
neg_features = extract_negative_features(neg_dir, width, height)

# Create positive and negative samples
pos_samples = create_samples(pos_features)
neg_samples = create_samples(neg_features)

# Create positive and negative labels
pos_labels = np.ones(len(pos_samples), dtype=int)
neg_labels = np.zeros(len(neg_samples), dtype=int)

samples = np.concatenate((pos_samples, neg_samples))
labels = np.concatenate((pos_labels, neg_labels))

# Write the positive samples to file
with open("pos.txt", "w") as f:
    for sample in pos_samples:
        f.write("pos/%s 1 0 0 %d %d\n" % ("img.jpg", width, height))

# Write the negative samples to file
with open("neg.txt", "w") as f:
    for filename in os.listdir(neg_dir):
        f.write("neg/%s\n" % filename)

# Create the positive samples vector file
os.system(
    "opencv_createsamples -info pos.txt -vec pos.vec -w %d -h %d" % (width, height)
)

# Train the Haar Cascade classifier
params = cv2.CascadeClassifier_Params()
params.numPos = 2400
params.numNeg = 2730
params.numStages = 20
params.featureType = cv2.HaarCascadeClassifier.LBP
params.minHitRate = 0.999
params.maxFalseAlarmRate = 0.5
params.w = width
params.h = height
params.numThreads = 2
params.precalcValBufSize = 5120
params.precalcIdxBufSize = 5120

classifier = cv2.CascadeClassifier()
classifier.train("pos.vec", "neg.txt", params=params)

# Save the trained classifier
classifier.save("face_classifier.xml")
