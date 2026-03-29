
# import sounddevice as sd
# import soundfile as sf
# import librosa
# import numpy as np
# import speech_recognition as sr
# from transformers import pipeline

# # IMPORT YOUR TEXT MODULE
# from text_stress_module import analyze_paragraph


# print("Loading voice emotion model...")

# emotion_classifier = pipeline(
#     "audio-classification",
#     model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# )

# print("Model loaded successfully!\n")


# # -----------------------------
# # RECORD VOICE (Press ENTER to stop)
# # -----------------------------
# def record_voice(filename="recorded_voice.wav", samplerate=16000):

#     print("\nRecording... Press ENTER to stop")

#     recording = []

#     def callback(indata, frames, time, status):
#         recording.append(indata.copy())

#     with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
#         input()  # Wait until ENTER pressed

#     audio = np.concatenate(recording, axis=0)
#     sf.write(filename, audio, samplerate)

#     print("Recording finished!")
#     return filename


# # -----------------------------
# # AUDIO FEATURES
# # -----------------------------
# def extract_audio_features(audio_path):

#     audio, sr = librosa.load(audio_path, sr=16000)

#     rms = np.mean(librosa.feature.rms(y=audio))
#     pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))

#     energy_score = min(rms * 100, 100)

#     return energy_score, pitch


# # -----------------------------
# # SPEECH → TEXT
# # -----------------------------
# def speech_to_text(audio_file):

#     recognizer = sr.Recognizer()

#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)

#     try:
#         text = recognizer.recognize_google(audio_data)
#     except:
#         text = ""

#     return text


# # -----------------------------
# # MAIN STRESS FUNCTION
# # -----------------------------
# def calculate_voice_stress(audio_path):

#     # VOICE EMOTION
#     emotion_result = emotion_classifier(audio_path)[0]
#     emotion = emotion_result["label"]

#     energy, pitch = extract_audio_features(audio_path)

#     # SPEECH → TEXT
#     recognized_text = speech_to_text(audio_path)

#     print("\nRecognized Text:", recognized_text)

#     # TEXT ANALYSIS
#     text_result = analyze_paragraph(recognized_text)

#     if text_result is not None:
#         text_level = text_result["stress_level"]
#         text_score = text_result["stress_intensity"]
#     else:
#         text_level = "Unknown"
#         text_score = 0

#     # VOICE SCORE
#     voice_score = 0

#     if emotion in ["angry", "fear", "sad"]:
#         voice_score += 40
#     elif emotion == "surprise":
#         voice_score += 25
#     else:
#         voice_score += 10

#     voice_score += energy * 0.5

#     # FINAL COMBINATION
#     final_score = (0.6 * text_score) + (0.4 * voice_score)
#     final_score = min(final_score, 100)

#     if final_score <= 20:
#         stress_level = "No Stress"
#     elif final_score <= 45:
#         stress_level = "Low Stress"
#     elif final_score <= 70:
#         stress_level = "Moderate Stress"
#     else:
#         stress_level = "High Stress"

#     return emotion, recognized_text, text_level, stress_level, int(final_score)


# # -----------------------------
# # RUN LOOP
# # -----------------------------
# while True:

#     user_input = input("\nPress ENTER to start recording (type 'exit' to stop): ")

#     if user_input.lower() == "exit":
#         break

#     audio_file = record_voice()

#     emotion, text, text_level, level, intensity = calculate_voice_stress(audio_file)

#     print("\n🎤 Emotion:", emotion)
#     print("📝 Text:", text)
#     print("🧠 Text Stress:", text_level)
#     print("🔥 Final Stress:", level)
#     print("⚡ Intensity:", intensity, "%")





# import sounddevice as sd
# import soundfile as sf
# import librosa
# import numpy as np
# import speech_recognition as sr
# from transformers import pipeline

# from text_stress_module import analyze_paragraph

# print("Loading voice emotion model...")

# emotion_classifier = pipeline(
#     "audio-classification",
#     model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# )

# print("Model loaded successfully!\n")


# # -----------------------------
# # RECORD VOICE (Press ENTER to stop)
# # -----------------------------
# def record_voice(filename="recorded_voice.wav", samplerate=16000):

#     print("\nRecording... Press ENTER to stop")

#     recording = []

#     def callback(indata, frames, time, status):
#         recording.append(indata.copy())

#     with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
#         input()

#     audio = np.concatenate(recording, axis=0)
#     sf.write(filename, audio, samplerate)

#     print("Recording finished!")
#     return filename


# # -----------------------------
# # AUDIO FEATURES
# # -----------------------------
# def extract_audio_features(audio_path):

#     audio, sr = librosa.load(audio_path, sr=16000)

#     rms = np.mean(librosa.feature.rms(y=audio))
#     if rms<0.01:
#         return None, None
#     pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))

#     energy_score = min(rms * 100, 100)

#     return energy_score, pitch


# # -----------------------------
# # SPEECH TO TEXT
# # -----------------------------
# def speech_to_text(audio_file):

#     recognizer = sr.Recognizer()

#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)

#     try:
#         text = recognizer.recognize_google(audio_data)
#     except:
#         text = ""

#     return text


# # -----------------------------
# # MAIN STRESS FUNCTION
# # -----------------------------
# def calculate_voice_stress(audio_path):

#     # 🎤 Voice emotion
#     emotion_result = emotion_classifier(audio_path)[0]
#     emotion = emotion_result["label"]

#     energy, pitch = extract_audio_features(audio_path)
#     if energy is None:
#         return None, None, "No Input", "No Input", 0

#     # 📝 Speech → Text
#     recognized_text = speech_to_text(audio_path)

#     print("\nRecognized Text:", recognized_text)

#     # 🧠 TEXT ANALYSIS
#     text_result = analyze_paragraph(recognized_text)

#     if text_result is not None:
#         text_level = text_result["stress_level"]
#         text_score = text_result["stress_intensity"]
#     else:
#         text_level = "Unknown"
#         text_score = 0

#     # 🎧 VOICE SCORE
#     voice_score = 0

#     if emotion in ["angry", "fear", "sad"]:
#         voice_score += 40
#     elif emotion == "surprise":
#         voice_score += 25
#     else:
#         voice_score += 10

#     # 🔥 STRONGER ENERGY IMPACT (UPDATED)
#     voice_score += energy * 1.2

#     # -----------------------------
#     # 🧠 SMART FUSION LOGIC
#     # -----------------------------

#     # 🚨 CONTRADICTION DETECTION
#     if text_score < 20 and voice_score > 50:
#         final_score = 50 + (voice_score * 0.5)

#     # 🎤 VOICE DOMINATES (if text calm)
#     elif text_score < 20:
#         final_score = (0.4 * text_score) + (0.6 * voice_score)

#     # 📝 TEXT DOMINATES (normal case)
#     else:
#         final_score = (0.6 * text_score) + (0.4 * voice_score)

#     final_score = min(final_score, 100)

#     # -----------------------------
#     # STRESS LEVEL
#     # -----------------------------
#     if final_score <= 20:
#         stress_level = "No Stress"
#     elif final_score <= 45:
#         stress_level = "Low Stress"
#     elif final_score <= 70:
#         stress_level = "Moderate Stress"
#     else:
#         stress_level = "High Stress"

#     return emotion, recognized_text, text_level, stress_level, int(final_score)


# # -----------------------------
# # RUN LOOP
# # -----------------------------
# while True:

#     user_input = input("\nPress ENTER to start recording (type 'exit' to stop): ")

#     if user_input.lower() == "exit":
#         break

#     audio_file = record_voice()

#     emotion, text, text_level, level, intensity = calculate_voice_stress(audio_file)
#     if level == "No Input":print("\n No voice detected, Please try again")
#     continue

#     print("\n🎤 Emotion:", emotion)
#     print("📝 Text:", text)
#     print("🧠 Text Stress:", text_level)
#     print("🔥 Final Stress:", level)
#     print("⚡ Intensity:", intensity, "%")
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import speech_recognition as sr
from transformers import pipeline

# ✅ IMPORT YOUR TEXT MODEL
from text_stress_module import analyze_paragraph

print("Loading voice emotion model...")

emotion_classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

# -----------------------------
# RECORD VOICE (START/STOP)
# -----------------------------
def record_voice(filename="recorded_voice.wav", samplerate=16000):

    print("\n🎙 Recording... Press ENTER to stop")

    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        input()  # wait until ENTER

    audio = np.concatenate(recording, axis=0)
    sf.write(filename, audio, samplerate)

    print("✅ Recording finished!")

    return filename


# -----------------------------
# SPEECH TO TEXT
# -----------------------------
def speech_to_text(audio_path):

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text.lower()

    except:
        return ""


# -----------------------------
# AUDIO FEATURES
# -----------------------------
def extract_audio_features(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    rms = np.mean(librosa.feature.rms(y=audio))

    # ✅ Silence detection
    if rms < 0.01:
        return None, None

    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))

    energy_score = min(rms * 100, 100)

    return energy_score, pitch


# -----------------------------
# MAIN VOICE STRESS FUNCTION
# -----------------------------
def calculate_voice_stress(audio_path):

    # 🎤 Speech → Text
    recognized_text = speech_to_text(audio_path)

    # 🎧 Voice features
    energy, pitch = extract_audio_features(audio_path)

    # ❗ TRUE NO INPUT CASE
    if (energy is None) and (recognized_text.strip() == ""):
        return None, None, "No Input", "No Input", 0

    # 🎭 Emotion from voice
    emotion_result = emotion_classifier(audio_path)[0]
    emotion = emotion_result["label"]

    # -----------------------------
    # TEXT ANALYSIS
    # -----------------------------
    if recognized_text.strip() != "":
        text_result = analyze_paragraph(recognized_text)

        if text_result:
            text_level = text_result["stress_level"]
            text_score = text_result["stress_intensity"]
        else:
            text_level = "Unknown"
            text_score = 0
    else:
        text_level = "Unknown"
        text_score = 0

    # -----------------------------
    # VOICE STRESS SCORE
    # -----------------------------
    voice_score = 0

    if emotion in ["angry", "fear", "sad"]:
        voice_score += 40
    elif emotion in ["surprise"]:
        voice_score += 25
    else:
        voice_score += 10

    if energy is not None:
        voice_score += energy * 0.5

    # -----------------------------
    # FINAL COMBINED SCORE
    # -----------------------------
    if text_score >= 80:
        final_score = text_score
    else:
        final_score = (0.4 * voice_score)+(0.6*text_score)

    # -----------------------------
    # LEVEL MAPPING
    # -----------------------------
    if final_score <= 20:
        level = "No Stress"
    elif final_score <= 45:
        level = "Low Stress"
    elif final_score <= 70:
        level = "Moderate Stress"
    else:
        level = "High Stress"

    return emotion, recognized_text, text_level, level, int(final_score)


# -----------------------------
# RUN LOOP
# -----------------------------
while True:

    user_input = input("\nPress ENTER to start recording (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        break

    audio_file = record_voice()

    emotion, text, text_level, level, intensity = calculate_voice_stress(audio_file)

    # ✅ HANDLE NO INPUT
    if level == "No Input":
        print("\n⚠️ No voice detected. Please try again.")
        continue

    # ✅ PRINT RESULTS
    print("\n🎤 Emotion:", emotion)
    print("📝 Text:", text)
    print("🧠 Text Stress:", text_level)
    print("🔥 Final Stress:", level)
    print("⚡ Intensity:", intensity, "%")