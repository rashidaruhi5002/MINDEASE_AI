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

print("✅ Model loaded!\n")


# -----------------------------
# 🎙 FIXED RECORDING (15 SEC)
# -----------------------------
def record_voice(filename="recorded_voice.wav", samplerate=16000, duration=15):

    print(f"\n🎙 Recording for {duration} seconds...")

    recording = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=1)

    sd.wait()

    sf.write(filename, recording, samplerate)

    print("✅ Recording finished!")

    return filename


# -----------------------------
# 🧠 SPEECH TO TEXT (CHUNKED)
# -----------------------------
def speech_to_text(audio_path):

    recognizer = sr.Recognizer()
    audio, sr_rate = librosa.load(audio_path, sr=16000)

    chunk_size = 5 * sr_rate  # 5 sec chunks
    full_text = ""

    for i in range(0, len(audio), chunk_size):

        chunk = audio[i:i + chunk_size]
        temp_file = "temp.wav"
        sf.write(temp_file, chunk, sr_rate)

        with sr.AudioFile(temp_file) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            full_text += " " + text
        except:
            continue

    return full_text.strip().lower()


# -----------------------------
# 🎧 AUDIO FEATURES
# -----------------------------
def extract_audio_features(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    rms = np.mean(librosa.feature.rms(y=audio))

    # ❗ silence detection
    if rms < 0.01:
        return None, None

    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    energy_score = min(rms * 100, 100)

    return energy_score, pitch


# -----------------------------
# 🔥 MAIN VOICE STRESS
# -----------------------------
def calculate_voice_stress(audio_path):

    # 🎤 text
    recognized_text = speech_to_text(audio_path)

    # 🎧 features
    energy, pitch = extract_audio_features(audio_path)

    if (energy is None) and (recognized_text.strip() == ""):
        return None, None, "No Input", "No Input", 0

    # 🎭 emotion
    emotion_result = emotion_classifier(audio_path)[0]
    emotion = emotion_result["label"]

    # -----------------------------
    # TEXT ANALYSIS
    # -----------------------------
    if recognized_text:
        text_result = analyze_paragraph(recognized_text)

        if text_result:
            text_score = text_result["stress_intensity"]
            text_level = text_result["stress_level"]
        else:
            text_score = 0
            text_level = "Unknown"
    else:
        text_score = 0
        text_level = "Unknown"

    # -----------------------------
    # VOICE SCORE
    # -----------------------------
    voice_score = 0

    if emotion in ["angry", "fear", "sad"]:
        voice_score += 40
    elif emotion == "surprise":
        voice_score += 25
    else:
        voice_score += 10

    if energy is not None:
        voice_score += energy * 0.5

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    if text_score >= 80:
        final_score = text_score
    else:
        final_score = (0.6 * text_score) + (0.4 * voice_score)

    final_score = min(final_score, 100)

    # -----------------------------
    # LEVEL
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
# ▶ RUN LOOP
# -----------------------------
while True:

    user_input = input("\nPress ENTER to start recording (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        break

    audio_file = record_voice()

    emotion, text, text_level, level, intensity = calculate_voice_stress(audio_file)

    if level == "No Input":
        print("\n⚠️ No voice detected. Try again.")
        continue

    print("\n🎤 Emotion:", emotion)
    print("📝 Text:", text)
    print("🧠 Text Stress:", text_level)
    print("🔥 Final Stress:", level)
    print("⚡ Intensity:", intensity, "%")