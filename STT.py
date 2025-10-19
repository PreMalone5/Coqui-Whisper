import numpy as np
import librosa
from scipy import signal
from faster_whisper import WhisperModel

AUDIO_PATH = "rec_1.wav"

# 1) Model: prefer large-v3. If CPU: try "int8_float32" (better than pure int8); if GPU: float16.
model = WhisperModel(
    "large-v3",
    device="cpu",            # "cuda" if you have GPU; else "cpu"
    compute_type="float32"    # GPU: "float16"; CPU best quality: "float32" (or "int8_float32" if RAM limited)
)

# 2) Load audio at 16 kHz mono
audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

# 3) Gentle cleanup: high-pass 50 Hz (optional low-pass ~7.9 kHz if very noisy)
nyq = 0.5 * sr
hp = 50 / nyq
b, a = signal.butter(2, hp, btype="highpass")
audio = signal.filtfilt(b, a, audio)

# Optional: light low-pass (comment out if original is already clean)
# lp = 7900 / nyq
# b2, a2 = signal.butter(2, lp, btype="lowpass")
# audio = signal.filtfilt(b2, a2, audio)

# Normalize
mx = np.max(np.abs(audio)) + 1e-9
audio = (audio / mx).astype(np.float32)

# 4) Transcribe with Arabic bias + anti-drift
segments, info = model.transcribe(
    audio,
    language="ar",
    beam_size=8,
    best_of=5,
    temperature=0.0,
    condition_on_previous_text=False,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300},
    initial_prompt="هذا تسجيل باللغة العربية المصرية."
)

print(f"lang_guess={info.language} p={info.language_probability:.3f} (forced ar)")
for s in segments:
    print(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}")
