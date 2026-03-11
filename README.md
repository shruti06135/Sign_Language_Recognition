# Sign Language Recognition using Deep Learning

A deep learning based system for **real-time sign language recognition** using hand and pose landmarks extracted with **MediaPipe** and a **BiLSTM + Attention neural network**.

The system processes video input, extracts skeletal landmarks, converts them into temporal sequences, and predicts the corresponding sign language word using a trained neural network model.

---

# Project Overview

Sign language recognition enables communication between deaf or hard-of-hearing individuals and those unfamiliar with sign language.

This project implements a **temporal deep learning pipeline** that:

1. Extracts human pose and hand landmarks from video using MediaPipe
2. Converts them into structured feature sequences
3. Uses a **Bidirectional LSTM with Attention mechanism** to model temporal motion patterns
4. Predicts the corresponding sign language word
5. Performs **real-time inference from webcam input**

---

# Model Architecture

The final model uses a **Bidirectional LSTM with Attention** architecture designed to capture temporal dependencies in gesture sequences.

### Input Representation

Each video is converted into a sequence of:

```
30 frames × 225 features
```

Features include:

* Pose landmarks
* Left hand landmarks
* Right hand landmarks

Each landmark contributes **(x, y, z) coordinates**.

---

### Network Structure

```
Input (30 × 225)

Bidirectional LSTM (48 units)
Batch Normalization
Dropout

Bidirectional LSTM (48 units)
Batch Normalization
Dropout

Attention Layer

Global Average Pooling

Dense (128 units)
Dropout

Output Layer (Softmax – 262 classes)
```

---

# Dataset Pipeline

The dataset preparation pipeline converts raw sign language videos into structured sequences used for training.

```
Raw Videos
    ↓
Frame Sampling
    ↓
MediaPipe Landmark Extraction
    ↓
Sequence Generation
    ↓
Normalization
    ↓
Train/Test Split
    ↓
Model Training
```

---

# Dataset Format

Each processed sample contains:

```
Sequence Length: 30 frames
Feature Dimension: 225 landmarks
```

Dataset shape:

```
X_train : (3406, 30, 225)
X_test  : (852, 30, 225)
```

Number of sign classes:

```
262 words
```

---

# Training Details

Training configuration:

```
Optimizer: Adam
Learning Rate: 0.0005
Loss Function: Categorical Crossentropy (Label Smoothing)
Batch Size: 32
Epochs: 80
```

Additional techniques used:

* Feature normalization
* Batch normalization
* Dropout regularization
* Early stopping
* Model checkpointing

---

# Model Performance

Final validation performance:

```
Training Accuracy: ~91%
Validation Accuracy: ~81%
```

The model successfully learns temporal gesture patterns and generalizes well to unseen sequences.

---

# Real-Time Prediction Pipeline

The real-time recognition system operates as follows:

```
Webcam Input
      ↓
MediaPipe Landmark Detection
      ↓
Landmark Feature Extraction
      ↓
30-Frame Sequence Buffer
      ↓
Sequence Normalization
      ↓
Trained BiLSTM + Attention Model
      ↓
Softmax Prediction
      ↓
Word Output
```

Predicted class indices are converted to words using:

```
data/label_map.json
```

---

# Repository Structure

```
sign_language_project

app/                Frontend application code
inference/          Real-time prediction scripts
models/             Trained neural network models

preprocessing/      Dataset generation pipeline
    build_sequences.py
    landmark_extractor.py
    frame_sampler.py
    normalize_dataset.py

training/           Model training scripts
    train_model_attention_improved.py

data/
    label_map.json  Class index to word mapping

requirements.txt
.gitignore
README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running Real-Time Prediction

To run the real-time sign recognition system:

```
python inference/predict_realtime.py
```

The system will:

1. Start the webcam
2. Extract MediaPipe landmarks
3. Maintain a rolling 30-frame buffer
4. Predict the corresponding sign language word

---

# Technologies Used

* Python
* TensorFlow / Keras
* MediaPipe
* OpenCV
* NumPy
* Scikit-learn

---

# Future Improvements

Possible extensions of this project include:

* Sentence-level sign language recognition
* Transformer-based temporal models
* Gesture segmentation
* Larger multi-signer datasets
* Deployment via web or mobile applications

---

# Acknowledgements

This project was developed as part of an academic deep learning project focusing on **sign language understanding using sequence models**.

---

# License

This project is intended for academic and research purposes.
