import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load data
landmarks_file = 'data/landmarks.csv'
labels_file = 'data/labels.csv'

landmarks = pd.read_csv(landmarks_file, header=None)
labels = pd.read_csv(labels_file, header=None)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels.values.ravel())
num_classes = len(np.unique(labels_encoded))
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    landmarks.values, labels_categorical, test_size=0.2, random_state=42
)

# Reshape data for CNN
X_train = X_train.reshape(-1, 21, 3, 1)  # 21 landmarks, 3 coordinates (x, y, z)
X_test = X_test.reshape(-1, 21, 3, 1)

# Save preprocessed data
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

# Save label encoder classes
np.save('data/classes.npy', label_encoder.classes_)

print("Data preprocessing completed and saved successfully.")
