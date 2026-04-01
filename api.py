"""
api.py — Flask API layer for MindEase AI
-----------------------------------------
Exposes endpoints:
    POST /analyze-text
        Input  (JSON): { "text": "..." }
        Output (JSON): { "stress_level": <int 0–100>, "emotion": "<string>" }

    POST /analyze-text-simple
        Input  (JSON): { "diaryText": "..." }
        Output (JSON): { "textStress": <int 0–100> }
        Lightweight keyword-based analysis — no ML models, < 5 ms response.

    POST /analyze-face
        Input  (multipart/form-data): image file in field "image"
        Output (JSON): { "stress_level": <int 0–100>, "emotion": "<string>" }

    POST /analyze-voice
        Input  (multipart/form-data): audio file in field "audio"
        Output (JSON): { "stress_level": <int 0–100>, "emotion": "<string>" }

This file wraps text_stress_finalmodule.py, face_stress_module.py,
and voice_stress_module.py.  No model code is modified.
"""

import os
import sys
import types
import tempfile
import subprocess
import importlib.util

import imageio_ffmpeg

# Make imageio_ffmpeg's bundled ffmpeg binary discoverable as "ffmpeg" on PATH.
# The bundled binary has a versioned name (e.g. ffmpeg-win-x86_64-v7.1.exe) so
# libraries that search for plain "ffmpeg" (like HuggingFace transformers) won't
# find it.  We create a shim directory with a copy named "ffmpeg.exe".
import shutil as _shutil

_ffmpeg_real = imageio_ffmpeg.get_ffmpeg_exe()
_ffmpeg_shim_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_shim")
_ffmpeg_shim = os.path.join(_ffmpeg_shim_dir, "ffmpeg.exe")

if not os.path.exists(_ffmpeg_shim):
    os.makedirs(_ffmpeg_shim_dir, exist_ok=True)
    _shutil.copy2(_ffmpeg_real, _ffmpeg_shim)

if _ffmpeg_shim_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_shim_dir + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, request, jsonify
from deepface import DeepFace

# Import ONLY the paragraph analyser from the final text model.
# The module loads its three HuggingFace pipelines at import time.
from text_stress_finalmodule import analyze_paragraph

# Import the emotion-to-stress mapping from the face model.
# We do NOT call analyze_face() (it uses webcam/GUI).
from face_stress_module import get_stress_from_emotion

# ---------------------------------------------------------------------------
# Import calculate_voice_stress from voice_stress_module WITHOUT executing
# the module-level while-True interactive loop.  We load the module source,
# strip out the top-level loop, and exec only the definitions.
# ---------------------------------------------------------------------------
def _load_voice_module():
    """Return the calculate_voice_stress function from voice_stress_module."""
    module_path = os.path.join(os.path.dirname(__file__), "voice_stress_module.py")
    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Remove the interactive while-True block that lives at module level.
    # It starts with "while True:" and runs to end-of-file.
    marker = "\nwhile True:"
    idx = source.rfind(marker)
    if idx != -1:
        source = source[:idx]

    mod = types.ModuleType("voice_stress_module")
    mod.__file__ = module_path
    # Ensure the module's directory is on sys.path so its own imports resolve.
    mod_dir = os.path.dirname(module_path)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    exec(compile(source, module_path, "exec"), mod.__dict__)  # noqa: S102
    return mod.calculate_voice_stress


calculate_voice_stress = _load_voice_module()


# ---------------------------------------------------------------------------
# Audio format helper — convert any audio to 16 kHz mono PCM WAV so that
# speech_recognition (inside the voice model) can read it.
# ---------------------------------------------------------------------------
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def _convert_to_pcm_wav(input_path: str) -> str:
    """Convert *input_path* to a 16 kHz, mono, 16-bit PCM WAV temp file.
    Returns the path to the new WAV file (caller must delete it)."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        _FFMPEG_EXE,
        "-y",              # overwrite output
        "-i", input_path,  # input file
        "-ar", "16000",    # 16 kHz sample rate
        "-ac", "1",        # mono
        "-sample_fmt", "s16",  # 16-bit signed PCM
        "-f", "wav",       # force WAV container
        wav_path
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
    )
    if result.returncode != 0:
        # Clean up on failure
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise RuntimeError(
            f"ffmpeg conversion failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:500]}"
        )
    return wav_path


app = Flask(__name__)


# ---------------------------------------------------------------------------
# POST /analyze-text
# ---------------------------------------------------------------------------
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    """
    Accepts JSON body: { "text": "some user text" }
    Returns:          { "stress_level": <int>, "emotion": "<string>" }
    """

    # ── 1. Parse request body ──────────────────────────────────────────────
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid input. Request body must be valid JSON."
        }), 400

    text = data.get("text", "")

    # ── 2. Validate input ──────────────────────────────────────────────────
    if not isinstance(text, str):
        return jsonify({
            "error": "Invalid input. 'text' must be a string."
        }), 400

    text = text.strip()

    if not text:
        return jsonify({
            "error": "Empty text. Please provide a non-empty 'text' field."
        }), 400

    # ── 3. Run the text model ──────────────────────────────────────────────
    try:
        result = analyze_paragraph(text)
    except Exception as e:
        return jsonify({
            "error": "Server error while analysing text.",
            "detail": str(e)
        }), 500

    # ── 4. Handle model returning None (text too short) ────────────────────
    if result is None:
        return jsonify({
            "error": (
                "Text is too short for analysis. "
                "Please provide at least one sentence with 3 or more words."
            )
        }), 422

    # ── 5. Return structured response ──────────────────────────────────────
    return jsonify({
        "stress_level": result["stress_intensity"],   # int  0–100
        "emotion":      result["dominant_emotion"]    # str  e.g. "joy"
    }), 200


# ---------------------------------------------------------------------------
# POST /analyze-face
# ---------------------------------------------------------------------------
@app.route("/analyze-face", methods=["POST"])
def analyze_face_endpoint():
    """
    Accepts multipart/form-data with an image in the "image" field.
    Returns: { "stress_level": <int 0-100>, "emotion": "<string>" }
    """

    # ── DEBUG: log incoming request details ──────────────────────────────
    print("Headers:", request.headers)
    print("Files:", request.files)
    print("Form:", request.form)

    # ── 1. Check that an image file was uploaded ──────────────────────────
    file = request.files.get('image')

    if file is None:
        return jsonify({"error": "No image uploaded. Please send an image in the 'image' field."}), 400

    if file.filename == '':
        return jsonify({"error": "No image selected."}), 400

    # ── 2. Validate file type ─────────────────────────────────────────────
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        return jsonify({
            "error": (
                f"Invalid image format '{ext}'. "
                f"Allowed formats: {', '.join(sorted(allowed_extensions))}"
            )
        }), 400

    # ── 3. Save image to a temp file ──────────────────────────────────────
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        file.save(tmp_path)

        # ── 4. Run DeepFace analysis ─────────────────────────────────────
        result = DeepFace.analyze(
            img_path=tmp_path,
            actions=['emotion'],
            enforce_detection=False
        )

        # Handle result safely
        if isinstance(result, list):
            result = result[0]

        emotion = result.get('dominant_emotion', 'neutral')

        # Use the face model's mapping function to get stress score.
        stress_score = get_stress_from_emotion(emotion)

        return jsonify({
            "stress_level": stress_score,
            "emotion":      emotion
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Face analysis failed",
            "detail": str(e)
        }), 500

    finally:
        # ── 5. Clean up temp file ─────────────────────────────────────────
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# POST /analyze-voice  (with chunk processing for long audio)
# ---------------------------------------------------------------------------
import librosa
import soundfile as sf
import numpy as np
from collections import Counter

# Duration of each audio chunk in seconds.
_CHUNK_DURATION_SEC = 10


def _split_wav_into_chunks(wav_path, chunk_duration=_CHUNK_DURATION_SEC):
    """Split a WAV file into a list of temp-file paths, each ≤ chunk_duration
    seconds long.  Returns (list_of_paths, sample_rate).  Caller must delete
    the temp files afterwards."""
    audio, sr_rate = librosa.load(wav_path, sr=16000, mono=True)
    total_samples = len(audio)
    chunk_samples = chunk_duration * sr_rate  # 10 s × 16 000 = 160 000 samples

    # Short audio — no splitting needed.
    if total_samples <= chunk_samples:
        return [wav_path], sr_rate

    chunk_paths = []
    for start in range(0, total_samples, chunk_samples):
        segment = audio[start : start + chunk_samples]

        # Skip very short trailing fragments (< 1 s).
        if len(segment) < sr_rate:
            continue

        fd, chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(chunk_path, segment, sr_rate)
        chunk_paths.append(chunk_path)

    return chunk_paths, sr_rate


@app.route("/analyze-voice", methods=["POST"])
def analyze_voice_endpoint():
    """
    Accepts multipart/form-data with an audio file in the "audio" field.
    Returns: { "stress_level": <int 0-100>, "emotion": "<string>" }

    Long audio files are automatically split into ~10-second chunks.
    Each chunk is analysed independently; results are then aggregated:
      • stress_level  →  average across all valid chunks
      • emotion       →  most frequent emotion (majority vote)
    """

    # ── 1. Check that an audio file was uploaded ──────────────────────────
    file = request.files.get('audio')

    if file is None:
        return jsonify({"error": "No audio uploaded. Please send an audio file in the 'audio' field."}), 400

    if file.filename == '':
        return jsonify({"error": "No audio file selected."}), 400

    # ── 2. Validate file type ─────────────────────────────────────────────
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        return jsonify({
            "error": (
                f"Invalid audio format '{ext}'. "
                f"Allowed formats: {', '.join(sorted(allowed_extensions))}"
            )
        }), 400

    # ── 3. Save uploaded audio to a temp file ──────────────────────────────
    tmp_path = None
    wav_path = None
    chunk_paths = []
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        file.save(tmp_path)

        # ── 4. Convert to PCM WAV (speech_recognition only reads PCM WAV) ─
        wav_path = _convert_to_pcm_wav(tmp_path)

        # ── 5. Split audio into chunks ────────────────────────────────────
        chunk_paths, _ = _split_wav_into_chunks(wav_path)

        # ── 6. Analyse each chunk ─────────────────────────────────────────
        stress_scores = []
        emotions = []

        for chunk_path in chunk_paths:
            result = calculate_voice_stress(chunk_path)
            emotion, recognized_text, text_level, stress_label, final_score = result

            # Skip silent / "No Input" chunks.
            if stress_label == "No Input":
                continue

            stress_scores.append(final_score)
            emotions.append(emotion)

        # ── 7. Handle case where ALL chunks were silent ───────────────────
        if not stress_scores:
            return jsonify({
                "error": "No voice detected in the audio. Please try again with a clearer recording."
            }), 400

        # ── 8. Aggregate results ──────────────────────────────────────────
        avg_stress = int(round(sum(stress_scores) / len(stress_scores)))
        majority_emotion = Counter(emotions).most_common(1)[0][0]

        return jsonify({
            "stress_level": avg_stress,         # int 0-100
            "emotion":      majority_emotion    # str e.g. "angry"
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Voice analysis failed.",
            "detail": str(e)
        }), 500

    finally:
        # ── 9. Clean up ALL temp files ────────────────────────────────────
        # chunk_paths may include wav_path itself (short audio), so use a set.
        all_temps = {tmp_path, wav_path} | set(chunk_paths)
        for p in all_temps:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# POST /analyze-text-simple   (keyword-based, no ML models)
# ---------------------------------------------------------------------------
# Curated word sets — kept at module level so they are built once.
_NEGATIVE_WORDS = frozenset([
    # anxiety / fear
    "stressed", "stress", "anxious", "anxiety", "worried", "worry", "nervous",
    "panic", "scared", "afraid", "terrified", "dread", "uneasy", "tense",
    "overwhelmed", "overthinking", "restless", "insecure",
    # sadness / despair
    "sad", "depressed", "depression", "hopeless", "helpless", "miserable",
    "lonely", "alone", "crying", "cry", "tears", "grief", "heartbroken",
    "suffering", "pain", "painful", "hurt", "broken", "empty", "numb",
    # anger / frustration
    "angry", "furious", "rage", "frustrated", "irritated", "annoyed",
    "hate", "hatred", "resentment", "bitter", "hostile",
    # fatigue / burnout
    "exhausted", "tired", "burnout", "drained", "fatigued", "sleepless",
    "insomnia", "overworked",
    # general negativity
    "terrible", "awful", "horrible", "worst", "bad", "ugly", "fail",
    "failure", "useless", "worthless", "pathetic", "disgusted", "sick",
    "trapped", "stuck", "lost", "confused", "disappointed", "regret",
    "ashamed", "guilty", "jealous", "envious",
    # crisis
    "suicide", "suicidal", "die", "death", "kill", "harm", "self-harm",
])

_POSITIVE_WORDS = frozenset([
    # happiness / joy
    "happy", "joy", "joyful", "cheerful", "glad", "delighted", "excited",
    "thrilled", "wonderful", "amazing", "fantastic", "great", "awesome",
    "good", "excellent", "brilliant", "beautiful",
    # calm / peace
    "calm", "peaceful", "relaxed", "serene", "tranquil", "content",
    "comfortable", "safe", "secure", "balanced", "steady",
    # gratitude / love
    "grateful", "thankful", "blessed", "love", "loved", "caring",
    "kind", "compassionate", "warm", "supportive", "appreciated",
    # hope / motivation
    "hopeful", "optimistic", "motivated", "inspired", "confident",
    "strong", "proud", "determined", "energetic", "refreshed",
    # general positivity
    "fun", "enjoy", "smile", "laugh", "better", "best", "success",
    "progress", "achieved", "accomplished", "improved", "healing",
])

# High-impact crisis words that push stress much higher per occurrence.
_CRISIS_WORDS = frozenset([
    "suicide", "suicidal", "self-harm", "kill", "die", "death",
])


def _keyword_stress(text: str) -> int:
    """Return a stress percentage 0-100 from simple keyword analysis.

    Algorithm
    ---------
    1.  Start at a neutral baseline of 30.
    2.  For every negative keyword found, add  +5  (crisis words add +15).
    3.  For every positive keyword found, subtract -4.
    4.  Clamp the result to [0, 100].

    The scoring is intentionally conservative: short texts with one or two
    negative words land in the 35-45 range (mild), while diary entries
    loaded with crisis language quickly reach 90+.
    """
    words = set(text.lower().split())  # unique tokens (fast O(n))

    neg_hits = words & _NEGATIVE_WORDS
    pos_hits = words & _POSITIVE_WORDS
    crisis_hits = words & _CRISIS_WORDS

    # Baseline
    score = 30.0

    # Regular negative / positive adjustments
    score += len(neg_hits) * 5
    score -= len(pos_hits) * 4

    # Crisis words get an extra bump (they are already counted as negative)
    score += len(crisis_hits) * 10

    return int(max(0, min(100, round(score))))


@app.route("/analyze-text-simple", methods=["POST"])
def analyze_text_simple():
    """
    Lightweight, keyword-based diary stress analysis.

    Accepts JSON body:  { "diaryText": "some user diary text" }
    Returns:            { "textStress": <int 0-100> }

    • Uses no ML models — responds in < 5 ms.
    • Empty / missing text safely returns { "textStress": 0 }.
    """

    # ── 1. Parse request body ─────────────────────────────────────────────
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid input. Request body must be valid JSON."
        }), 400

    diary_text = data.get("diaryText", "")

    # ── 2. Validate input ─────────────────────────────────────────────────
    if not isinstance(diary_text, str):
        return jsonify({
            "error": "Invalid input. 'diaryText' must be a string."
        }), 400

    diary_text = diary_text.strip()

    # Empty text → 0 stress (safe default)
    if not diary_text:
        return jsonify({"textStress": 0}), 200

    # ── 3. Compute keyword-based stress ───────────────────────────────────
    stress = _keyword_stress(diary_text)

    return jsonify({"textStress": stress}), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run on 0.0.0.0 so it is reachable from mobile/other devices on LAN.
    # use_reloader=False prevents the watchdog from restarting the server
    # when TensorFlow/DeepFace writes cache files during analysis.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
