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
import random
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
# Response Enrichment — human-like, stress-tier-aware messages
# ---------------------------------------------------------------------------
# Each tier has MULTIPLE options per emotion so responses feel varied and real.
# The system picks one at random on every request.
# ---------------------------------------------------------------------------

_TIER_LOW = "low"          # 0–30   → positive / motivational
_TIER_MILD = "mild"        # 31–60  → light calming
_TIER_HIGH = "high"        # 61–80  → breathing / relaxation
_TIER_CRITICAL = "critical" # 81–100 → strong calming + urgency


def _get_tier(stress: int) -> str:
    if stress <= 30:
        return _TIER_LOW
    if stress <= 60:
        return _TIER_MILD
    if stress <= 80:
        return _TIER_HIGH
    return _TIER_CRITICAL


# ── Messages ──────────────────────────────────────────────────────────────
# Keyed by (tier, emotion). Falls back to (tier, "default") when the
# specific emotion isn't listed.

_MESSAGES = {
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TIER: LOW  (0–30)  —  positive, encouraging, warm                 ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    (_TIER_LOW, "happy"): [
        "You're radiating good energy right now — keep riding this wave! 🌊",
        "That smile isn't just on your face, it's in your whole vibe. Keep going!",
        "You're in a really good headspace. Take a moment to appreciate that.",
        "Happiness looks good on you — and your stress is super low. Well done!",
    ],
    (_TIER_LOW, "joy"): [
        "There's genuine joy in your words — that's something to celebrate! 🎉",
        "Your positivity is practically contagious right now. Keep doing what you're doing.",
        "Moments like this are worth savoring. You're doing great.",
    ],
    (_TIER_LOW, "neutral"): [
        "You seem calm and centered — that's a healthy place to be.",
        "Steady and balanced. Your mind is in a good resting state right now.",
        "Nothing dramatic, nothing heavy — just a nice, grounded moment.",
        "You're in a stable headspace. A great foundation to build on.",
    ],
    (_TIER_LOW, "surprise"): [
        "Something caught your attention! And the good news — your stress is low.",
        "A little surprise can be energizing. Your overall state looks great.",
    ],
    (_TIER_LOW, "default"): [
        "You're doing really well right now — your stress is impressively low.",
        "Whatever you've been doing, it's working. Keep it up! 💪",
        "Your mind seems clear and light. This is a great moment to enjoy.",
        "Low stress detected — you're in a healthy zone. Stay mindful of this feeling.",
        "You're in a calm, positive space. Take a breath and appreciate it.",
    ],

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TIER: MILD  (31–60)  —  light calming, gentle check-in            ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    (_TIER_MILD, "sad"): [
        "It looks like something's weighing on you. That's okay — it's human to feel this way.",
        "I can sense some sadness. You don't have to push through it alone.",
        "A little heaviness in your words — would a short break help you reset?",
        "Feeling down is part of life, but you don't have to sit in it silently.",
    ],
    (_TIER_MILD, "angry"): [
        "Sounds like something frustrated you. A quick pause can help clear the fog.",
        "I notice some tension. Before it builds, let's try to soften it a little.",
        "Anger often carries a message. Once the heat cools, you'll see it more clearly.",
    ],
    (_TIER_MILD, "fear"): [
        "There's some anxiety present, but it's manageable. Let's keep it that way.",
        "A little worry is normal — the key is not letting it snowball.",
        "I see some nervousness. You're safe right now; let's focus on that.",
    ],
    (_TIER_MILD, "disgust"): [
        "Something's bothering you, and that's completely valid.",
        "That unsettled feeling can pass — let's give your mind a gentle reset.",
    ],
    (_TIER_MILD, "neutral"): [
        "You're mostly fine, but I notice a hint of stress. Nothing major — just worth noting.",
        "Things seem okay on the surface. Check in with yourself — how do you really feel?",
        "Moderate range — not bad, but a little self-care could go a long way.",
    ],
    (_TIER_MILD, "default"): [
        "Your stress is in a moderate range — nothing alarming, but worth addressing.",
        "I'm picking up some mild tension. A short break might be all you need.",
        "You're not in a bad place, but a quick mental reset could help you feel even better.",
        "Moderate stress detected. This is a good time to pause and recalibrate.",
        "Things are manageable, but don't ignore that subtle tension. Small steps help.",
    ],

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TIER: HIGH  (61–80)  —  breathing, relaxation, grounding          ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    (_TIER_HIGH, "angry"): [
        "Your frustration is real, and it's pushing your stress up. Let's bring it down together.",
        "Anger is intense energy — let's redirect it before it drains you.",
        "I can feel the heat in your words. Let's cool things down with a breathing exercise.",
    ],
    (_TIER_HIGH, "sad"): [
        "This feels heavy, and I want you to know — it's okay to not be okay right now.",
        "Sadness at this level needs attention, not avoidance. Let's take care of you.",
        "You're carrying a lot. Let's start by grounding yourself in this moment.",
    ],
    (_TIER_HIGH, "fear"): [
        "Anxiety is high. Your body is on alert — let's tell it that you're safe.",
        "That feeling of dread? It's your nervous system overreacting. Let's calm it.",
        "Fear is loud right now, but you can quiet it. Start with your breathing.",
    ],
    (_TIER_HIGH, "disgust"): [
        "Something is deeply unsettling you. Let's step back and reset.",
        "That strong reaction is valid — but let's not let it consume you.",
    ],
    (_TIER_HIGH, "default"): [
        "Your stress is elevated. I'd strongly recommend pausing what you're doing.",
        "This is a good time to step away for a few minutes and breathe. You need it.",
        "High stress detected — your mind and body are asking for a break.",
        "You're under significant pressure. Let's try a grounding technique right now.",
        "This level of stress can affect your thinking. Let's bring you back to center.",
    ],

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  TIER: CRITICAL  (81–100)  —  strong calming, urgency              ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    (_TIER_CRITICAL, "angry"): [
        "Your anger is at a critical level. Please stop what you're doing and focus on your breath.",
        "This level of rage can hurt you. Please — pause, close your eyes, breathe slowly.",
        "I hear you. The anger is overwhelming right now. Let's ground you before anything else.",
    ],
    (_TIER_CRITICAL, "sad"): [
        "This is a deeply painful moment. You don't have to face it alone — please reach out to someone you trust.",
        "I can feel how heavy this is. Please be gentle with yourself right now. You matter.",
        "Extreme sadness can make everything feel impossible. One breath at a time. You're still here.",
    ],
    (_TIER_CRITICAL, "fear"): [
        "Your anxiety is extremely high. Let's focus on one simple thing: 5 things you can see around you.",
        "The fear feels unbearable, but your body CAN calm down. Let me guide you through grounding.",
        "Panic-level anxiety detected. You are safe. Let's slow everything down right now.",
    ],
    (_TIER_CRITICAL, "default"): [
        "Your stress is critically high. Please take immediate action — step away, breathe, reach out.",
        "This level of stress is not sustainable. Your wellbeing comes first. Please pause everything.",
        "I'm genuinely concerned. Please stop, breathe deeply, and consider talking to someone you trust.",
        "Critical stress detected. Nothing you're doing right now is more important than your health.",
        "Your mind and body are overwhelmed. Please — close your eyes, take 10 slow breaths. Right now.",
    ],
}


# ── Suggestions ───────────────────────────────────────────────────────────
# Actionable, specific advice per tier.

_SUGGESTIONS = {
    _TIER_LOW: [
        "Write down three things you're grateful for today — it anchors positivity.",
        "Share your good mood with someone. Send a kind text or call a friend.",
        "Take a 10-minute walk outside and soak in your surroundings.",
        "Use this calm moment to plan something you've been putting off.",
        "Try journaling about what made today good. It builds emotional awareness.",
        "Set a small, achievable goal for the rest of your day — momentum is powerful.",
        "Listen to your favorite uplifting song and let yourself enjoy it fully.",
        "Drink a glass of water and stretch for 2 minutes. Small wins matter.",
    ],
    _TIER_MILD: [
        "Step away from your screen for 5 minutes. Look at something far away to relax your eyes and mind.",
        "Make yourself a warm drink — tea, coffee, anything comforting — and sip it slowly.",
        "Write down what's on your mind. Getting it out of your head often reduces the weight.",
        "Put on some calm background music (lo-fi, nature sounds) for the next 15 minutes.",
        "Try the 5-4-3-2-1 method: name 5 things you see, 4 you hear, 3 you can touch, 2 you smell, 1 you taste.",
        "Call or text someone who makes you feel safe. Connection is a powerful stress reliever.",
        "Take a short walk, even if it's just around your room. Movement shifts your mental state.",
        "Do a quick brain dump: set a timer for 3 minutes and write everything on your mind without filtering.",
    ],
    _TIER_HIGH: [
        "Close your eyes and do the 4-7-8 breathing technique: inhale for 4s, hold for 7s, exhale for 8s. Repeat 4 times.",
        "Place both feet flat on the ground. Feel the floor beneath you. You are here. You are safe.",
        "Splash cold water on your face — it activates your dive reflex and calms your nervous system.",
        "Hold an ice cube in your hand for 30 seconds. The sensation interrupts the stress spiral.",
        "Step outside if you can. Fresh air and a change of scenery can reset your nervous system.",
        "Tense every muscle in your body for 5 seconds, then release all at once. Feel the difference.",
        "Talk to someone right now — a friend, family member, or helpline. Don't carry this alone.",
        "Lie down if possible, place a hand on your chest and one on your belly. Breathe until your belly hand rises.",
    ],
    _TIER_CRITICAL: [
        "STOP everything. Sit or lie down. Close your eyes. Breathe in for 4 counts, out for 6 counts. Do this 10 times.",
        "Call or message someone you trust immediately. You need human connection right now.",
        "If you're having thoughts of self-harm, please contact a crisis helpline. You are not alone.",
        "Place one hand on your heart and one on your stomach. Feel your heartbeat. You are alive and you matter.",
        "Go to a quiet space. Remove all distractions. Focus only on the sound of your own breathing.",
        "Splash very cold water on your face and wrists. This physiologically forces your heart rate down.",
        "If possible, step outside and stand barefoot on grass or cool ground. Feel the earth beneath you.",
        "Ask someone to sit with you — even silently. Presence alone can make a difference right now.",
    ],
}


# ── Exercises ─────────────────────────────────────────────────────────────
# Meaningful, guided exercises per tier.

_EXERCISES = {
    _TIER_LOW: [
        "Gratitude Scan: Close your eyes, take 3 slow breaths, and mentally list 5 things that made you smile recently.",
        "Mindful Sip: Make a warm drink, hold the cup with both hands, and focus entirely on the taste and warmth for 2 minutes.",
        "Joy Anchoring: Think of your happiest recent memory. Replay it in full detail — the sounds, smells, and feelings.",
        "Body Appreciation: Starting from your toes, mentally thank each part of your body for what it does for you.",
        "Positive Visualization: Spend 2 minutes imagining your ideal tomorrow. What does a perfect day look like?",
        "Soundtrack Reset: Play a song that makes you feel alive. Close your eyes and just listen — nothing else.",
    ],
    _TIER_MILD: [
        "Box Breathing: Inhale for 4 seconds, hold for 4, exhale for 4, hold for 4. Repeat 5 rounds. This steadies your nervous system.",
        "Progressive Relaxation: Starting from your feet, tense each muscle group for 5 seconds then release. Work up to your face.",
        "5-4-3-2-1 Grounding: Name 5 things you see, 4 things you can touch, 3 things you hear, 2 things you smell, 1 thing you taste.",
        "Gentle Stretch Sequence: Roll your neck slowly 5 times each direction, then shrug your shoulders to your ears and drop them. Repeat 3 times.",
        "Worry Parking: Write each worry on a sticky note. Place them in a jar or box. Tell yourself: 'These can wait 30 minutes.'",
        "Nature Minute: Look out a window or step outside. Spend 60 seconds noticing natural details — clouds, leaves, light.",
    ],
    _TIER_HIGH: [
        "4-7-8 Calm Breathing: Breathe in through your nose for 4 seconds, hold for 7 seconds, exhale slowly through your mouth for 8 seconds. Repeat 4 times.",
        "Body Scan Meditation: Lie down. Starting from the top of your head, slowly scan down to your toes. Notice tension. Breathe into tight spots for 3 breaths each.",
        "Cold Reset: Run cold water over your wrists for 30 seconds while taking slow, deep breaths. The cold activates your calming nervous system.",
        "Butterfly Hug: Cross your arms over your chest, hands on shoulders. Gently tap left, right, left, right — slowly — for 1 minute while breathing deeply.",
        "Grounding Anchor: Press your feet firmly into the floor. Push your back against your chair. Feel gravity holding you. Say 'I am here. I am safe.' 5 times.",
        "Tension Release Wave: Clench both fists as hard as you can for 10 seconds. Release. Feel the warmth flow into your fingers. Repeat with your whole body.",
    ],
    _TIER_CRITICAL: [
        "Emergency Breathing: Breathe in for 4 counts, hold for 2, breathe out for 6 counts. Keep going for at least 2 full minutes. Focus ONLY on counting.",
        "Dive Reflex Reset: Fill a bowl with cold water. Hold your breath and submerge your face for 15-30 seconds. This triggers your body's natural calming response.",
        "5-Senses Anchor: Right now — name 5 things you can see. 4 things you can hear. 3 things you can touch. 2 things you can smell. 1 thing you can taste. Stay present.",
        "Safe Place Visualization: Close your eyes. Imagine a place where you feel completely safe. Describe it in detail — what you see, hear, feel. Stay there for 3 minutes.",
        "Physiological Sigh: Take two quick inhales through your nose (sniff-sniff), then one long, slow exhale through your mouth. Repeat 5 times. This is scientifically proven to reduce stress fast.",
        "Human Connection: Call or text one person right now. You don't have to explain everything — just say 'I'm having a hard time.' Let someone be there for you.",
    ],
}


def _enrich_response(stress_level: int, emotion: str) -> dict:
    """Build the enriched response fields: message, suggestion, exercise.

    The original stress_level and emotion are passed through unchanged.
    Three new fields are added using human-like, stress-tier-aware content.
    """
    tier = _get_tier(stress_level)
    emotion_lower = emotion.lower().strip()

    # ── Message: try emotion-specific, fall back to default ─────────────
    msg_pool = _MESSAGES.get((tier, emotion_lower)) or _MESSAGES[(tier, "default")]
    message = random.choice(msg_pool)

    # ── Suggestion & Exercise: tier-based ───────────────────────────────
    suggestion = random.choice(_SUGGESTIONS[tier])
    exercise = random.choice(_EXERCISES[tier])

    return {
        "stress_score": stress_level,
        "emotion":      emotion,
        "message":      message,
        "suggestion":   suggestion,
        "exercise":     exercise,
    }



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
    stress = result["stress_intensity"]
    emotion = result["dominant_emotion"]
    response = _enrich_response(stress, emotion)
    # Keep original field for backward compatibility
    response["stress_level"] = stress
    return jsonify(response), 200


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

        response = _enrich_response(stress_score, emotion)
        # Keep original field for backward compatibility
        response["stress_level"] = stress_score
        return jsonify(response), 200

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

        response = _enrich_response(avg_stress, majority_emotion)
        # Keep original field for backward compatibility
        response["stress_level"] = avg_stress
        return jsonify(response), 200

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
