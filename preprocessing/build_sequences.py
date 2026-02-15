import os
import numpy as np
from tqdm import tqdm

from preprocessing.landmark_extractor import process_video

# Paths
VIDEO_PATHS_FILE = "data/video_paths.npy"
LABELS_FILE = "data/labels.npy"
SEQUENCES_DIR = "data/sequences"
SEQUENCE_LABELS_FILE = "data/sequence_labels.npy"

# Create sequences directory
os.makedirs(SEQUENCES_DIR, exist_ok=True)

# Load metadata
video_paths = np.load(VIDEO_PATHS_FILE, allow_pickle=True)
labels = np.load(LABELS_FILE)

sequence_labels = []

print("Starting sequence generation...")

for idx in tqdm(range(len(video_paths))):
    save_path = os.path.join(SEQUENCES_DIR, f"{idx:05d}.npy")

    # Skip if already processed (resume support)
    if os.path.exists(save_path):
        sequence_labels.append(labels[idx])
        continue

    video_path = video_paths[idx]

    try:
        sequence = process_video(video_path)

        if sequence is None:
            print(f"Skipping corrupted video: {video_path}")
            continue

        np.save(save_path, sequence)
        sequence_labels.append(labels[idx])

    except Exception as e:
        print(f"Error processing {video_path}")
        print(e)
        continue

# Save aligned labels
np.save(SEQUENCE_LABELS_FILE, np.array(sequence_labels))

print("Sequence generation complete.")
print("Total sequences saved:", len(sequence_labels))