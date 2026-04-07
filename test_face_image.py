# from deepface import DeepFace
# import random

# image_path = "test_images/img test 1.jpg"

# print("Analyzing Image...")

# result = DeepFace.analyze(img_path=image_path, actions=['emotion'])

# emotion = result[0]['dominant_emotion']

# print("Detected Emotion:", emotion)

# # Emotion → Stress mapping

# if emotion in ["happy", "neutral"]:
#     stress = random.randint(0,20)
#     level = "No Stress"

# elif emotion in ["surprise"]:
#     stress = random.randint(20,45)
#     level = "Low Stress"

# elif emotion in ["sad"]:
#     stress = random.randint(50,70)
#     level = "Moderate Stress"

# else:
#     stress = random.randint(70,100)
#     level = "High Stress"

# print("Stress Level:", level)
# print("Stress Score:", stress,"%")


from deepface import DeepFace
import os

folder = "test_images"

print("Testing all images...\n")

for file in os.listdir(folder):

    if file.endswith(".jpg") or file.endswith(".png"):

        image_path = os.path.join(folder, file)

        print(f"\nAnalyzing {file}...")

        import cv2
        frame = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            img_path=frame_rgb,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='retinaface'
        )

        if isinstance(result, list):
            result = result[0]
        
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']

        print("Detected Emotion:", dominant_emotion)

        # 🔥 Better stress calculation (NO RANDOM)

        stress_score = (
            emotions.get("angry", 0) * 0.9 +
            emotions.get("fear", 0) * 0.9 +
            emotions.get("sad", 0) * 0.8 +
            emotions.get("disgust", 0) * 0.7 +
            emotions.get("surprise", 0) * 0.4 +
            emotions.get("neutral", 0) * 0.2 +
            emotions.get("happy", 0) * 0.1
        )

        stress_score = int(stress_score)

        # Stress level classification
        if stress_score <= 20:
            level = "No Stress"

        elif stress_score <= 45:
            level = "Low Stress"

        elif stress_score <= 70:
            level = "Moderate Stress"

        else:
            level = "High Stress"

        print("Stress Score:", stress_score, "%")
        print("Stress Level:", level)