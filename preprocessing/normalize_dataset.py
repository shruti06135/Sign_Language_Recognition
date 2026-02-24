import numpy as np
import os

DATA_DIR = "data/processed"

print("Loading datasets...")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Original shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


def normalize_sequences(X):
    """
    Normalize each video sequence by subtracting
    the mean landmark value across frames.
    """
    # Compute mean of each sequence across frames
    mean = np.mean(X, axis=1, keepdims=True)

    # Subtract mean
    X_norm = X - mean

    return X_norm


print("\nNormalizing sequences...")

X_train_norm = normalize_sequences(X_train)
X_test_norm = normalize_sequences(X_test)


print("Saving normalized datasets...")

np.save(os.path.join(DATA_DIR, "X_train_norm.npy"), X_train_norm)
np.save(os.path.join(DATA_DIR, "X_test_norm.npy"), X_test_norm)

# Labels remain same
np.save(os.path.join(DATA_DIR, "y_train_norm.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test_norm.npy"), y_test)


print("\nNormalized dataset saved!")

print("New files created:")
print("X_train_norm.npy")
print("X_test_norm.npy")
print("y_train_norm.npy")
print("y_test_norm.npy")