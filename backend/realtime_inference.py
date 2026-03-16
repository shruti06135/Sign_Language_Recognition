import cv2
import numpy as np
import mediapipe as mp
import json
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/best_model_attention_improved.h5")

# Load label mapping (class index -> word)
with open("data/label_map.json", "r") as f:
    label_map = json.load(f)  # {word: idx}

index_to_word = {idx: word for word, idx in label_map.items()}

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Sequence buffer
sequence = []

# Prediction history for stabilization
predictions = []

# Confidence threshold
THRESHOLD = 0.8

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])

# Webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        # Convert color
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        # Extract landmarks
        keypoints = extract_landmarks(results)

        sequence.append(keypoints)

        # Keep last 30 frames
        if len(sequence) > 30:
            sequence.pop(0)

        # Prediction
        if len(sequence) == 30:

            sequence_np = np.array(sequence)  # shape (30, 225)

            # Normalize sequence exactly like training:
            # subtract per-sequence mean across frames, no std division
            mean = np.mean(sequence_np, axis=0, keepdims=True)
            sequence_np = sequence_np - mean

            input_data = np.expand_dims(sequence_np, axis=0)

            prediction = model.predict(input_data)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class])

            word = index_to_word.get(predicted_class, f"UNK_{predicted_class}")

            print("Prediction:", word)
            print("Confidence:", confidence)

            predictions.append(predicted_class)

            if len(predictions) > 10:
                predictions.pop(0)

            # Majority vote stabilization
            if confidence > THRESHOLD:

                word = index_to_word.get(predicted_class, f"UNK_{predicted_class}")

                cv2.putText(
                    image,
                    f'{word} ({confidence:.2f})',
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2,
                    cv2.LINE_AA
                )

        # Display
        cv2.imshow("Sign Language Recognition", image)

        # Exit key
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()