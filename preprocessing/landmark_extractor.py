import cv2
import numpy as np
import mediapipe as mp
from preprocessing.frame_sampler import sample_frames

SEQUENCE_LENGTH = 30

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    landmarks = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0] * 33 * 3)

    # Left Hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0] * 21 * 3)

    # Right Hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0] * 21 * 3)

    return np.array(landmarks)


def process_video(video_path):
    frames = sample_frames(video_path)

    if frames is None:
        return None

    sequence = []

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        for frame in frames:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            landmarks = extract_landmarks(results)
            sequence.append(landmarks)

    return np.array(sequence)