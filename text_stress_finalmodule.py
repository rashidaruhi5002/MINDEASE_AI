# from transformers import pipeline
# import numpy as np
# import re

# print("Loading models... (first run may take a few minutes)")

# # Emotion detection model
# emotion_model = pipeline(
#     "text-classification",
#     model="SamLowe/roberta-base-go_emotions",
#     top_k=1
# )

# # Sentiment detection model
# sentiment_model = pipeline(
#     "text-classification",
#     model="cardiffnlp/twitter-roberta-base-sentiment",
#     top_k=1
# )

# # Mental health / suicidality model
# mental_model = pipeline(
#     "text-classification",
#     model="sentinet/suicidality",
#     top_k=1
# )

# print("Models loaded successfully!\n")


# # -----------------------------
# # Emotion → Stress score
# # -----------------------------
# def emotion_score(emotion):

#     mapping = {
#         "joy": 5,
#         "optimism": 10,
#         "neutral": 10,

#         "surprise": 15,

#         "sadness": 45,
#         "fear": 40,
#         "disappointment": 40,

#         "anger": 70,
#         "grief": 80,
#         "despair": 90
#     }

#     return mapping.get(emotion.lower(), 30)

# def convert_sentiment(label):

#     mapping = {
#         "LABEL_0": "negative",
#         "LABEL_1": "neutral",
#         "LABEL_2": "positive"
#     }

#     return mapping.get(label, label)
# # -----------------------------
# # Sentiment → Stress score
# # -----------------------------
# def sentiment_score(sentiment):

#     mapping = {
#         "positive": 5,
#         "neutral": 30,
#         "negative": 65
#     }

#     return mapping.get(sentiment.lower(), 40)


# # -----------------------------
# # Mental health score
# # -----------------------------
# def mental_score(label):

#     mapping = {
#         "LABEL_0": 10,   # safe
#         "LABEL_1": 95    # suicide risk
#     }

#     return mapping.get(label, 40)


# # -----------------------------
# # Sentence stress calculation
# # -----------------------------
# def calculate_stress(text):

#     emotion_result = emotion_model(text)[0][0]
#     emotion = emotion_result["label"]
#     e_score = emotion_score(emotion)

#     sentiment_result = sentiment_model(text)[0][0]
#     sentiment = convert_sentiment(sentiment_result["label"])
#     s_score = sentiment_score(sentiment)

#     mental_result = mental_model(text)[0][0]
#     mental_label = mental_result["label"]
#     m_score = mental_score(mental_label)

#    # Suicide override
#     if mental_label == "LABEL_1":
#         final_score = 95

#     # Calm / happy override
#     elif sentiment.lower() == "positive" and emotion.lower() in ["joy","optimism","neutral"]:
#         final_score = 5

#     else:
#         final_score = np.mean([e_score, s_score])

#     # Stress level mapping
#     if final_score <= 20:
#         level = "No Stress"

#     elif final_score <= 45:
#         level = "Low Stress"

#     elif final_score <= 70:
#         level = "Moderate Stress"

#     else:
#         level = "High Stress"

#     return {
#         "emotion": emotion,
#         "sentiment": sentiment,
#         "mental_signal": mental_label,
#         "stress_level": level,
#         "stress_intensity": int(final_score)
#     }


# # -----------------------------
# # Paragraph analyzer
# # -----------------------------
# def analyze_paragraph(text):

#     sentences = re.split(r'[.!?]', text)

#     scores = []
#     emotions = []

#     for sentence in sentences:

#         sentence = sentence.strip()

#         if len(sentence.split()) < 3:
#             continue

#         result = calculate_stress(sentence)

#         scores.append(result["stress_intensity"])
#         emotions.append(result["emotion"])

#     if len(scores) == 0:
#         return None

#     final_score = sum(scores) / len(scores)

#     if final_score <= 20:
#         level = "No Stress"

#     elif final_score <= 45:
#         level = "Low Stress"

#     elif final_score <= 70:
#         level = "Moderate Stress"

#     else:
#         level = "High Stress"

#     dominant_emotion = max(set(emotions), key=emotions.count)

#     return {
#         "stress_level": level,
#         "stress_intensity": int(final_score),
#         "dominant_emotion": dominant_emotion
#     }


# # -----------------------------
# # Continuous testing
# # -----------------------------
# while True:

#     text = input("\nEnter text (type 'exit' to stop): ")

#     if text.lower() == "exit":
#         break

#     result = analyze_paragraph(text)

#     if result is None:
#         print("\nText too short for analysis.")
#         continue

#     print("\nDominant Emotion:", result["dominant_emotion"])
#     print("Final Stress Level:", result["stress_level"])
#     print("Stress Intensity:", result["stress_intensity"], "%")
from transformers import pipeline
import numpy as np
import re

print("Loading models... (first run may take a few minutes)")

# Emotion detection model
emotion_model = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1
)

# Sentiment detection model
sentiment_model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    top_k=1
)

# Mental health / suicidality model
mental_model = pipeline(
    "text-classification",
    model="sentinet/suicidality",
    top_k=1
)

print("Models loaded successfully!\n")


# -----------------------------
# Emotion → Stress score
# -----------------------------
def emotion_score(emotion):

    mapping = {
        "joy": 5,
        "optimism": 10,
        "neutral": 10,

        "surprise": 15,

        "sadness": 45,
        "fear": 40,
        "disappointment": 40,

        "anger": 70,
        "grief": 80,
        "despair": 90
    }

    return mapping.get(emotion.lower(), 30)

def convert_sentiment(label):

    mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    return mapping.get(label, label)
# -----------------------------
# Sentiment → Stress score
# -----------------------------
def sentiment_score(sentiment):

    mapping = {
        "positive": 5,
        "neutral": 30,
        "negative": 65
    }

    return mapping.get(sentiment.lower(), 40)


# -----------------------------
# Mental health score
# -----------------------------
def mental_score(label):

    mapping = {
        "LABEL_0": 10,   # safe
        "LABEL_1": 95    # suicide risk
    }

    return mapping.get(label, 40)


# -----------------------------
# Sentence stress calculation
# -----------------------------
def calculate_stress(text):

    emotion_result = emotion_model(text)[0][0]
    emotion = emotion_result["label"]
    e_score = emotion_score(emotion)

    sentiment_result = sentiment_model(text)[0][0]
    sentiment = convert_sentiment(sentiment_result["label"])
    s_score = sentiment_score(sentiment)

    mental_result = mental_model(text)[0][0]
    mental_label = mental_result["label"]
    m_score = mental_score(mental_label)

   # Suicide override
    if mental_label == "LABEL_1":
        final_score = 95

    # Calm / happy override
    elif sentiment.lower() == "positive" and emotion.lower() in ["joy","optimism","neutral"]:
        final_score = 5

    else:
        final_score = np.mean([e_score, s_score])

    # Stress level mapping
    if final_score <= 20:
        level = "No Stress"

    elif final_score <= 45:
        level = "Low Stress"

    elif final_score <= 70:
        level = "Moderate Stress"

    else:
        level = "High Stress"

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "mental_signal": mental_label,
        "stress_level": level,
        "stress_intensity": int(final_score)
    }


# -----------------------------
# Paragraph analyzer
# -----------------------------
def analyze_paragraph(text):

    sentences = re.split(r'[.!?]', text)

    scores = []
    emotions = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) < 3:
            continue

        result = calculate_stress(sentence)

        scores.append(result["stress_intensity"])
        emotions.append(result["emotion"])

    if len(scores) == 0:
        return None

    final_score = sum(scores) / len(scores)

    if final_score <= 20:
        level = "No Stress"

    elif final_score <= 45:
        level = "Low Stress"

    elif final_score <= 70:
        level = "Moderate Stress"

    else:
        level = "High Stress"

    dominant_emotion = max(set(emotions), key=emotions.count)

    return {
        "stress_level": level,
        "stress_intensity": int(final_score),
        "dominant_emotion": dominant_emotion
    }


# -----------------------------
# Continuous testing
# -----------------------------
if __name__ == "__main__":

    while True:

        text = input("\nEnter text (type 'exit' to stop): ")

        if text.lower() == "exit":
            break

        result = analyze_paragraph(text)

        if result is None:
            print("\nText too short for analysis.")
            continue

        print("\nDominant Emotion:", result["dominant_emotion"])
        print("Final Stress Level:", result["stress_level"])
        print("Stress Intensity:", result["stress_intensity"], "%")