
import cv2
from deepface import DeepFace
import time
import numpy as np

# ---------------------------
# Emotion → Stress Mapping (IMPROVED)
# ---------------------------
def get_stress_from_emotion(emotion):
    mapping = {
        "happy": 5,
        "neutral": 20,
        "surprise": 40,
        "sad": 65,       # 🔥 increased
        "fear": 80,
        "angry": 90,
        "disgust": 75
    }
    return mapping.get(emotion, 30)


print("Starting Face Stress Analysis (10 seconds)...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

start_time = time.time()
stress_values = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Camera error")
        break

    cv2.imshow("Face Stress Detection", frame)

    # ---------------------------
    # EMOTION DETECTION (NO RESTRICTIONS)
    # ---------------------------
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']
        stress = get_stress_from_emotion(emotion)

        stress_values.append(stress)

        print(f"Emotion: {emotion} | Stress: {stress}%")

    except:
        pass

    # ---------------------------
    # STOP AFTER 10 SECONDS
    # ---------------------------
    if time.time() - start_time > 10:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# ---------------------------
# FINAL RESULT
# ---------------------------
if len(stress_values) > 0:

    stress_intensity = sum(stress_values) // len(stress_values)

    if stress_intensity < 20:
        level = "No Stress"
    elif stress_intensity < 40:
        level = "Low Stress"
    elif stress_intensity < 60:   # 🔥 changed from 65 → 60
        level = "Moderate Stress"
    else:
        level = "High Stress"

    print("\n✅ Face Stress Analysis Complete")
    print("Stress Intensity:", stress_intensity, "%")
    print("Stress Level:", level)

else:
    print("❌ No face detected properly")
