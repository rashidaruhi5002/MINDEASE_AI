import cv2
from deepface import DeepFace
import time

# ---------------------------
# Emotion → Stress Mapping
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
# MAIN LIVE FUNCTION
# ---------------------------
def analyze_face_live():

    print("🎥 Starting LIVE Face Stress Analysis...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    history = []  # 🔥 for smoothing

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Camera error")
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']
            stress = get_stress_from_emotion(emotion)

            # 🔁 SMOOTHING
            history.append(stress)
            if len(history) > 5:
                history.pop(0)

            final_stress = int(sum(history) / len(history))

            # ---------------------------
            # LEVEL
            # ---------------------------
            if final_stress < 20:
                level = "No Stress"
                color = (0, 255, 0)
            elif final_stress < 40:
                level = "Low Stress"
                color = (0, 255, 255)
            elif final_stress < 60:
                level = "Moderate Stress"
                color = (0, 165, 255)
            else:
                level = "High Stress"
                color = (0, 0, 255)

            # ---------------------------
            # DISPLAY ON SCREEN 🔥
            # ---------------------------
            cv2.putText(frame, f"Emotion: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, f"Stress: {final_stress}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, f"Level: {level}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except:
            pass

        cv2.imshow("Live Face Stress Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    analyze_face_live()