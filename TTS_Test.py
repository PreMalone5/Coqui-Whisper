import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
add_safe_globals([XttsConfig])  # allowlist the config class used inside the checkpoint
from TTS.api import TTS

# Get device
device = "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="EGTTS-V0.1/speaker_reference.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="EGTTS-V0.1/speaker_reference.wav", language="en", file_path="output.wav")
