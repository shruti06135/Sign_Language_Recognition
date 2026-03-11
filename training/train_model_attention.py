import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Bidirectional, Attention, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

# Load normalized dataset
X_train = np.load("data/processed/X_train_norm.npy")
X_test = np.load("data/processed/X_test_norm.npy")

y_train = np.load("data/processed/y_train_norm.npy")
y_test = np.load("data/processed/y_test_norm.npy")

print("Dataset shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Convert labels
num_classes = len(np.unique(y_train))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Input
inputs = Input(shape=(30,225))

# BiLSTM layers
x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

# Attention layer
attention = Attention()([x, x])

# Pooling
x = GlobalAveragePooling1D()(attention)

# Dense layers
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best model
checkpoint = ModelCheckpoint(
    "models/best_model_attention.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint]
)

# Save final model
model.save("models/final_model_attention.h5")