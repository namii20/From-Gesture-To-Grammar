# train_model.py ka code :
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 2: Load Alphabet Image Data
DATA_DIR = "C:/Users/Namita/Desktop/ASL_RECOGNITION_PROJECT/asl_alphabet_train"
IMG_SIZE = 64

labels = sorted(os.listdir(DATA_DIR))
print("Classes used:", labels)

X = []
y = []

# Step 3: Load & preprocess images
for idx, label in enumerate(labels):
    folder = os.path.join(DATA_DIR, label)
    image_files = os.listdir(folder)[:1000]  # Limit for faster training
    print(f"🔹 Loading {len(image_files)} images from class {label}")
    for img_name in tqdm(image_files):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)

# Convert to numpy arrays
X = np.array(X, dtype='float32') / 255.0
y = to_categorical(y, num_classes=len(labels))
print("✅ Total images loaded:", len(X))

# Step 4: Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')  # Output layer
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_val, y_val))

# Step 7: Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Save Model
model.save("asl_cnn_model.h5")