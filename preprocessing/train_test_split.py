import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Original dataset:")
print("X:", X.shape)
print("y:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nAfter split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Save splits
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print("\nTrain/Test dataset saved.")