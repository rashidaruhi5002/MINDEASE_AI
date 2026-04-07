import os
import cv2
import time
import logging
from deepface import DeepFace

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# Global buffer for smoothing
stress_buffer = []

# PART 2: RESTORE MAPPING
STRESS_MAPPING = {
    "happy": 15,
    "neutral": 35,
    "surprise": 45,
    "sad": 65,
    "fear": 75,
    "angry": 85,
    "disgust": 70
}

def analyze_face(image_path):
    """
    Restored DeepFace analysis system.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return {"stress_level": 35, "emotion": "neutral"}
    
    try:
        # STEP 1: CONVERT TO RGB AND USE DEEPFACE
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            frame_rgb,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Safely handle result list or dict
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']
        
        # STEP 2: RESTORE MAPPING
        stress = STRESS_MAPPING.get(emotion, 35)
        
        # STEP 3: ADD SMOOTHING
        global stress_buffer
        stress_buffer.append(stress)
        if len(stress_buffer) > 5:
            stress_buffer.pop(0)
            
        final_stress = sum(stress_buffer) / len(stress_buffer)
        
        # STEP 4: RETURN FORMAT
        return {
            "stress_level": int(final_stress),
            "emotion": emotion
        }
    except Exception:
        return {"stress_level": 35, "emotion": "neutral"}

def analyze_face_single():
    """
    Captures a frame from the camera and processes it using the stable old system.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"stress_level": 35, "emotion": "neutral"}

    # Stabilize
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        return {"stress_level": 35, "emotion": "neutral"}

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            frame_rgb,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='retinaface'
        )

        # Safely handle result list or dict
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']
        stress = STRESS_MAPPING.get(emotion, 35)
        
        global stress_buffer
        stress_buffer.append(stress)
        if len(stress_buffer) > 5:
            stress_buffer.pop(0)
            
        final_stress = sum(stress_buffer) / len(stress_buffer)
        
        return {
            "stress_level": int(final_stress),
            "emotion": emotion
        }
    except Exception:
        return {"stress_level": 35, "emotion": "neutral"}

if __name__ == "__main__":
    print(analyze_face_single())