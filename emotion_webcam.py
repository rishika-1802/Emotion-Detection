# =======================================
# emotion_webcam.py
# =======================================
import cv2
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime

# ================================
# 1. Load Trained Model
# ================================
model = tf.keras.models.load_model("emotion_cnn_model.h5")

# Emotion labels (same as FER-2013 classes)
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

# ================================
# 2. Setup Face Detector
# ================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================================
# 3. CSV Logger Setup
# ================================
csv_file = open("emotion_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Emotion", "Confidence"])

# ================================
# 4. Webcam Stream
# ================================
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw face bounding box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Preprocess face ROI
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48,48))
        face_normalized = face_resized.astype("float32") / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=-1)  # add channel
        face_reshaped = np.expand_dims(face_reshaped, axis=0)     # add batch

        # Predict emotion
        preds = model.predict(face_reshaped, verbose=0)
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]
        confidence = np.max(preds)

        # Display emotion
        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Log into CSV
        csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, f"{confidence:.4f}"])

    # Show video
    cv2.imshow("Real-Time Emotion Detection - Élodie 💕", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
