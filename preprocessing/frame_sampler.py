import cv2
import numpy as np

SEQUENCE_LENGTH = 30


def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)

    cap.release()

    total_frames = len(frames)

    if total_frames == 0:
        return None

    # Case 1: More frames than required → sample evenly
    if total_frames > SEQUENCE_LENGTH:
        indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)
        sampled_frames = [frames[i] for i in indices]

    # Case 2: Less frames → pad with last frame
    else:
        sampled_frames = frames.copy()
        while len(sampled_frames) < SEQUENCE_LENGTH:
            sampled_frames.append(frames[-1])

    return np.array(sampled_frames)