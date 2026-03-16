import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SEQUENCES_DIR = "data/sequences"
LABELS_FILE = "data/sequence_labels.npy"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------
# BUILD FULL DATASET (X, y)
# -----------------------------
print("Loading labels...")
labels = np.load(LABELS_FILE)

sequence_files = sorted(os.listdir(SEQUENCES_DIR))

print("Building X array...")
X = []

for file in tqdm(sequence_files):
    path = os.path.join(SEQUENCES_DIR, file)
    sequence = np.load(path)
    X.append(sequence)

X = np.array(X)
y = labels

print("Final full dataset shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(os.path.join(PROCESSED_DIR, "X.npy"), X)
np.save(os.path.join(PROCESSED_DIR, "y.npy"), y)

print("Full dataset saved to data/processed/")

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
print("\nSplitting into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nAfter split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("\nTrain/Test dataset saved to data/processed/")