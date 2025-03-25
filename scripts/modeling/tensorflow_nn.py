import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Reference Video: https://www.youtube.com/watch?v=F51Jx6pghHQ

# Read training data as df
training = pd.read_csv("train.csv")
training = training.drop(columns=["track_id"])
training = training.astype("float32")

# Read testing data as df
test = pd.read_csv("test.csv")
test = test.drop(columns=["track_id"])
test = test.astype("float32")

# Separate true labels from train/test
features = training.iloc[:, 0:-1].to_numpy()
labels = training.iloc[:, -1].to_numpy()
test_features = test.iloc[:, 0:-1].to_numpy()
test_labels = test.iloc[:, -1].to_numpy()

# Split training into train/val
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Feature Scaling - Note: very poor performance without scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_features = scaler.transform(test_features)

# Model Definition
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

# Configure model training
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_features, test_labels)

# Predict on the test set
test_predictions = (model.predict(test_features) > 0.5).astype("int32")

# Calculate and print the classification metrics
class_0_accuracy = accuracy_score(test_labels[test_labels == 0], test_predictions[test_labels == 0])
precision_0 = precision_score(test_labels, test_predictions, pos_label=0)
recall_0 = recall_score(test_labels, test_predictions, pos_label=0)
precision_1 = precision_score(test_labels, test_predictions, pos_label=1)
recall_1 = recall_score(test_labels, test_predictions, pos_label=1)
class_1_accuracy = accuracy_score(test_labels[test_labels == 1], test_predictions[test_labels == 1])

print(f"Overall Accuracy: {test_accuracy:.4f}")
print(f"Class 0 Accuracy: {class_0_accuracy:.4f}")
print(f"Class 0 Precision: {precision_0:.4f}")
print(f"Class 0 Recall: {recall_0:.4f}")

print(f"Class 1 Accuracy: {class_1_accuracy:.4f}")
print(f"Class 1 Precision: {precision_1:.4f}")
print(f"Class 1 Recall: {recall_1:.4f}")
