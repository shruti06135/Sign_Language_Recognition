import os
import numpy as np
from tqdm import tqdm

SEQUENCES_DIR = "data/sequences"
LABELS_FILE = "data/sequence_labels.npy"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

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

print("Final shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(os.path.join(PROCESSED_DIR, "X.npy"), X)
np.save(os.path.join(PROCESSED_DIR, "y.npy"), y)

print("Final dataset saved to data/processed/")