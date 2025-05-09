# realtime_mask_detector.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load face detector and trained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('mask_detector.h5')

# Define labels and colors
LABELS = ["Mask", "No Mask"]
COLORS = [(0, 255, 0), (0, 0, 255)]

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    mask_count = 0
    no_mask_count = 0

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0) / 255.0

        # Predict mask
        pred = model.predict(face)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"
        color = COLORS[0] if label == "Mask" else COLORS[1]

        # Count
        mask_count += 1 if label == "Mask" else 0
        no_mask_count += 1 if label == "No Mask" else 0

        # Draw box + label
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display count
    cv2.putText(frame, f"With Mask: {mask_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"No Mask: {no_mask_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("VisionMaskGuard - Real-Time Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
