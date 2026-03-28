import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from transformers import pipeline

print("Loading voice emotion model...")

emotion_classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

def record_voice(filename="recorded_voice.wav", duration=5, samplerate=16000):

    print("\nSpeak now...")

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()

    sf.write(filename, audio, samplerate)

    print("Recording finished!")

    return filename


def extract_audio_features(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    rms = np.mean(librosa.feature.rms(y=audio))
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))

    energy_score = min(rms * 100, 100)

    return energy_score, pitch


def calculate_voice_stress(audio_path):

    emotion_result = emotion_classifier(audio_path)[0]

    emotion = emotion_result["label"]

    energy, pitch = extract_audio_features(audio_path)

    stress_score = 0

    if emotion in ["angry", "fear", "sad"]:
        stress_score += 40

    elif emotion in ["surprise"]:
        stress_score += 25

    elif emotion in ["neutral"]:
        stress_score += 10

    stress_score += energy * 0.5

    stress_score = min(stress_score, 100)

    if stress_score <= 20:
        stress_level = "No Stress"

    elif stress_score <= 45:
        stress_level = "Low Stress"

    elif stress_score <= 70:
        stress_level = "Moderate Stress"

    else:
        stress_level = "High Stress"

    return emotion, stress_level, int(stress_score)


while True:

    user_input = input("\nPress ENTER to record voice (type exit to stop): ")

    if user_input.lower() == "exit":
        break

    audio_file = record_voice()

    emotion, level, intensity = calculate_voice_stress(audio_file)

    print("\nDetected Emotion:", emotion)
    print("Stress Level:", level)
    print("Stress Intensity:", intensity, "%")