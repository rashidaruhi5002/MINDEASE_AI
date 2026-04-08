import cv2
from deepface import DeepFace
import time

# ---------------------------
# Emotion → Stress Mapping (FRIEND'S ORIGINAL)
# ---------------------------
def get_stress_from_emotion(emotion):
    mapping = {
        "happy": 5,
        "neutral": 20,
        "surprise": 40,
        "sad": 65,
        "fear": 80,
        "angry": 90,
        "disgust": 75
    }
    return mapping.get(emotion, 30)


# ---------------------------
# IMAGE BASED (FOR API)
# ---------------------------
def analyze_face(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return {"stress_level": 30, "emotion": "neutral"}

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']
        stress = get_stress_from_emotion(emotion)

        return {
            "stress_level": stress,
            "emotion": emotion
        }

    except Exception:
        return {"stress_level": 30, "emotion": "neutral"}


# ---------------------------
# CAMERA BASED (FOR APP)
# ---------------------------
def analyze_face_single():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return {"stress_level": 30, "emotion": "neutral"}

    time.sleep(1)
    ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        return {"stress_level": 30, "emotion": "neutral"}

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']
        stress = get_stress_from_emotion(emotion)

        return {
            "stress_level": stress,
            "emotion": emotion
        }

    except Exception:
        return {"stress_level": 30, "emotion": "neutral"}


if __name__ == "__main__":
    print(analyze_face_single())