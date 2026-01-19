# asl_webcam.py  ka code :

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import string

# Load the trained model
model = load_model("asl_cnn_model.h5")

# Create label list: A-Z excluding J and Z
labels = list(string.ascii_uppercase)
labels.remove('J')
labels.remove('Z')

IMG_SIZE = 64  # Make sure this matches training size

# Start webcam
cap = cv2.VideoCapture(0)

print("🔁 Starting ASL Recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from webcam.")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)  # shape (1, IMG_SIZE, IMG_SIZE, 3)

    # Predict
    preds = model.predict(roi_input)[0]
    pred_index = np.argmax(preds)
    pred_label = labels[pred_index]
    confidence = preds[pred_index]

    # Display on frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"{pred_label} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Webcam", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()