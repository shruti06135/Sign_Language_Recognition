import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    Attention,
    GlobalAveragePooling1D,
    BatchNormalization
)

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Reproducibility
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

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

inputs = Input(shape=(30,225))

# First BiLSTM
x = Bidirectional(LSTM(48, return_sequences=True))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Second BiLSTM
x = Bidirectional(LSTM(48, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Attention
attention = Attention()([x, x])

# Pooling
x = GlobalAveragePooling1D()(attention)

# Dense layers
x = Dense(
    128,
    activation="relu",
    kernel_regularizer=l2(0.001)
)(x)

x = Dropout(0.4)(x)

outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

# -----------------------------
# COMPILE
# -----------------------------

optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

checkpoint = ModelCheckpoint(
    "models/best_model_attention_improved.h5",
    monitor="val_accuracy",
    save_best_only=True
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# Save final model 
model.save("models/final_model_attention_improved.h5")