import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("Dataset shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Convert labels to categorical
num_classes = len(np.unique(y_train))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build model
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(30,225)))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation="softmax"))

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best model
checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint]
)

# Save final model
model.save("models/final_model.h5")