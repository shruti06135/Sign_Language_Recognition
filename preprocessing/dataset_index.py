import os
import json
import numpy as np

DATA_PATH = "data/raw/INCLUDE"

video_extensions = (".mp4", ".mov", ".avi")

video_paths = []
labels = []
class_names = []

# Traverse categories
for category in os.listdir(DATA_PATH):
    category_path = os.path.join(DATA_PATH, category)

    if not os.path.isdir(category_path):
        continue

    # Inside category there is one subfolder
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        # Now inside this → word folders
        for word in os.listdir(subfolder_path):
            word_path = os.path.join(subfolder_path, word)

            if not os.path.isdir(word_path):
                continue

            # Clean word name (remove numbering like "1. ")
            clean_word = word.split(". ", 1)[-1] if ". " in word else word
            clean_word = clean_word.strip()

            if clean_word not in class_names:
                class_names.append(clean_word)

            # Collect video files
            for file in os.listdir(word_path):
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(word_path, file)
                    video_paths.append(full_path)
                    labels.append(clean_word)

# Sort class names for consistent indexing
class_names = sorted(class_names)

# Create label map
label_map = {name: idx for idx, name in enumerate(class_names)}

# Convert word labels to numeric
numeric_labels = [label_map[label] for label in labels]

# Save outputs
os.makedirs("data", exist_ok=True)

np.save("data/video_paths.npy", video_paths)
np.save("data/labels.npy", numeric_labels)

with open("data/label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("Total classes:", len(class_names))
print("Total videos:", len(video_paths))