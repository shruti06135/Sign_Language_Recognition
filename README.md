# ✋ Sign Language Recognition using Deep Learning

A deep learning system for **real-time sign language recognition** using **MediaPipe landmarks** and a **BiLSTM + Attention neural network**.

The system processes video input, extracts skeletal landmarks, converts them into temporal sequences, and predicts the corresponding **sign language word in real time**.

---

# 🚀 Features

✔ Real-time sign language prediction from webcam
✔ Landmark extraction using MediaPipe
✔ Temporal modeling using **Bidirectional LSTM**
✔ Attention mechanism for improved sequence understanding
✔ Data preprocessing and sequence generation pipeline
✔ Normalized landmark features for better training stability
✔ Modular project structure for easy integration with frontend/backend

---

# 🧠 Model Architecture

The final model uses a **BiLSTM + Attention architecture** to learn temporal patterns of hand and body movements.

### Input Representation

Each video is converted into:

```
30 frames × 225 features
```

Features include:

* Pose landmarks
* Left hand landmarks
* Right hand landmarks

Each landmark contributes **(x, y, z) coordinates**.

---

### Neural Network Structure

```
Input (30 × 225)

BiLSTM (48 units)
Batch Normalization
Dropout

BiLSTM (48 units)
Batch Normalization
Dropout

Attention Layer

Global Average Pooling

Dense (128)
Dropout

Softmax Output (262 classes)
```

---

# 📊 Dataset Pipeline

The dataset pipeline converts raw sign language videos into structured sequences.

```
Raw Videos
   ↓
Frame Sampling
   ↓
MediaPipe Landmark Extraction
   ↓
Sequence Generation
   ↓
Feature Normalization
   ↓
Train/Test Split
   ↓
Model Training
```

Dataset size used for training:

```
X_train : (3406, 30, 225)
X_test  : (852, 30, 225)
Classes : 262 sign words
```

---

# 📈 Model Performance

Training configuration:

```
Optimizer : Adam
Batch Size : 32
Epochs : 80
Loss : Categorical Crossentropy
```

Final performance:

```
Training Accuracy  ≈ 91%
Validation Accuracy ≈ 81%
```

The model learns temporal motion patterns effectively and generalizes well to unseen sequences.

---

# 🎥 Real-Time Prediction Pipeline

The real-time recognition system works as follows:

```
Webcam Input
     ↓
MediaPipe Landmark Detection
     ↓
Feature Extraction
     ↓
30 Frame Sequence Buffer
     ↓
Trained BiLSTM + Attention Model
     ↓
Softmax Prediction
     ↓
Predicted Word
```

Class predictions are mapped to words using:

```
data/label_map.json
```

---

# 📁 Project Structure

```
sign_language_project

app/                 Frontend components
inference/           Real-time prediction scripts

models/              Trained neural network models

preprocessing/
    frame_sampler.py
    landmark_extractor.py
    build_sequences.py
    normalize_dataset.py

training/
    train_model.py
    train_model_attention_improved.py

data/
    label_map.json

requirements.txt
.gitignore
README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶ Running Real-Time Prediction

```
python inference/predict_realtime.py
```

The system will:

1. Start the webcam
2. Extract MediaPipe landmarks
3. Maintain a 30-frame sequence buffer
4. Predict the sign language word

---

# 🛠 Technologies Used

* Python
* TensorFlow / Keras
* MediaPipe
* OpenCV
* NumPy
* Scikit-learn

---

# 🔮 Future Improvements

Possible future enhancements:

* Sentence-level sign language recognition
* Transformer-based sequence models
* Mobile / web deployment
* Larger multi-signer datasets
* Real-time translation to speech

---

# 👩‍💻 Contributors

Developed as part of a **Deep Learning project on Sign Language Understanding**.

---

# 📄 License

This project is intended for **academic and research purposes**.
