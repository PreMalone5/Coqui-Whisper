import sounddevice as sd
from scipy import signal
import librosa
import numpy as np
from faster_whisper import WhisperModel

model_size = "large-v2"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Load audio
audio_data, sample_rate = librosa.load("rec_1.wav", sr=None)

# Apply bandpass filter to reduce noise
nyquist = 0.5 * sample_rate
low = 100 / nyquist   # Wider low frequency range
high = 4000 / nyquist # Extended high frequency range
b, a = signal.butter(6, [low, high], btype='band')  # Higher order filter
filtered_audio = signal.filtfilt(b, a, audio_data)

# Convert to float32 to match ONNX model requirements
filtered_audio = filtered_audio.astype(np.float32)

# Play the denoised audio
print("Playing denoised audio...")
sd.play(filtered_audio, sample_rate)
sd.wait()  # Wait until the audio finishes playing

# Transcribe with sampling rate specified
segments, info = model.transcribe(
    filtered_audio,
    beam_size=5,
    vad_filter=True,  # Enable VAD filtering
    vad_parameters=dict(min_silence_duration_ms=500),
    temperature=0.0,  # Lower temperature for more consistent results
    best_of=5,  # Increase best_of parameter
    patience=2.0,  # Add patience for better decoding
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
