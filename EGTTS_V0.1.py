import os
import json
import torch
import torchaudio
import torch.nn as nn
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ========= Settings =========
MODEL_DIR = "EGTTS-V0.1"
CONFIG_FILE_PATH = os.path.join(MODEL_DIR, "config.json")
VOCAB_FILE_PATH  = os.path.join(MODEL_DIR, "vocab.json")
CHECKPOINT_DIR   = MODEL_DIR
SPEAKER_AUDIO_PATH = os.path.join(MODEL_DIR, "speaker_reference.wav")
LATENTS_CACHE = os.path.join(MODEL_DIR, "speaker_latents.pt")
OUTPUT_WAV = "egtss_output.wav"

# Decoding: slightly tighter (less work, steadier prosody)
DECODE_KW = dict(
    temperature=0.5,
    repetition_penalty=3.0,
    top_k=30,
    top_p=0.8
)

# Runtime knobs
ENABLE_QUANT = True         # dynamic quantization on CPU
ENABLE_COMPILE = False      # set True if using PyTorch>=2.3 and want to try torch.compile
NUM_THREADS = int(os.environ.get("PY_NUM_THREADS", "8"))   # tune to physical cores
NUM_INTEROP = int(os.environ.get("PY_NUM_INTEROP", "1"))

# ========= Fast CPU setup =========
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_INTEROP)
torch.backends.mkldnn.enabled = True  # usually on by default on x86
print(f"Threads: intra={NUM_THREADS}, interop={NUM_INTEROP}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ========= Speaker ref: ensure 16kHz mono (fast resample) =========
def ensure_16k_mono(path):
    waveform, sr = torchaudio.load(path)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        # Use a cheaper resampler (lower filter width) to speed up
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=16000, lowpass_filter_width=6
        )
        sr = 16000
        torchaudio.save(path, waveform, sr)  # overwrite only if changed
    return path

ensure_16k_mono(SPEAKER_AUDIO_PATH)

# ========= Load model =========
config = XttsConfig()
config.load_json(CONFIG_FILE_PATH)

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=CHECKPOINT_DIR,
    use_deepspeed=False,
    vocab_path=VOCAB_FILE_PATH
)

# If CPU: quantize linear/LSTM/GRU-heavy parts
if device == "cpu" and ENABLE_QUANT:
    # Try to quantize large submodules if exposed (failsafe: quantize the whole model)
    qtypes = {nn.Linear, nn.LSTM, nn.GRU}
    try:
        model = torch.quantization.quantize_dynamic(model, qtypes, dtype=torch.qint8)
        print("✅ Dynamic quantization applied (Linear/LSTM/GRU -> int8).")
    except Exception as e:
        print("⚠️ Quantization skipped:", repr(e))

# Optional compile (may help on some CPUs; small warm-up cost)
if ENABLE_COMPILE and device == "cpu":
    try:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
        print("✅ torch.compile enabled (inductor).")
    except Exception as e:
        print("⚠️ torch.compile skipped:", repr(e))

if device == "cuda":
    model.cuda()

# ========= Speaker latents (cached) =========
def load_or_compute_latents(model, cache_path, audio_path):
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        print("Loaded cached speaker latents.")
        return data["gpt_cond_latent"], data["speaker_embedding"]

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[audio_path])
    # move to CPU for caching regardless of compute device
    to_cpu = lambda x: x.detach().to("cpu") if torch.is_tensor(x) else x
    torch.save({
        "gpt_cond_latent": to_cpu(gpt_cond_latent),
        "speaker_embedding": to_cpu(speaker_embedding)
    }, cache_path)
    print("Latents computed and cached.")
    return gpt_cond_latent, speaker_embedding

gpt_cond_latent, speaker_embedding = load_or_compute_latents(model, LATENTS_CACHE, SPEAKER_AUDIO_PATH)

# ========= Inference =========
text = "يا جدعان قولنا 100 مرة بطلوا الخرا الى بتعملوه ده كل واحد يخليه فى حاله"
print("Synthesizing...")

with torch.inference_mode():
    out = model.inference(
        text=text,
        language="ar",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        **DECODE_KW
    )

# Normalize possible outputs
if isinstance(out, dict):
    audio = next((out[k] for k in ("audio","wav","samples","y_hat","tts_output") if k in out), None)
    sr_out = out.get("sample_rate", 24000)
elif isinstance(out, (list, tuple)) and len(out) > 0:
    audio = out[0]
    sr_out = 24000
else:

    raise ValueError(f"Unexpected inference return type: {type(out)}")

if audio is None:
    raise ValueError("No audio array found in model output.")

# Save (keep in torch to avoid extra numpy copies)
if not torch.is_tensor(audio):
    audio = torch.tensor(audio)

audio = audio.to(dtype=torch.float32).view(1, -1).clamp_(-1.0, 1.0)
torchaudio.save(OUTPUT_WAV, audio, sr_out)
dur = audio.shape[1] / float(sr_out)
print(f"✅ Speech saved as {OUTPUT_WAV} ({dur:.2f}s @ {sr_out} Hz)")
