---
widget:
- src: sample.flac
  output:
    text: صباح الخير.
license: other
license_name: coqui-public-model-license
language: ar
base_model: coqui/XTTS-v2
pipeline_tag: text-to-speech
---
# EGTTS V0.1
EGTTS V0.1 is a cutting-edge text-to-speech (TTS) model specifically designed for Egyptian Arabic. Built on the XTTS v2 architecture, it transforms written Egyptian Arabic text into natural-sounding speech, enabling seamless communication in various applications such as voice assistants, educational tools, and chatbots.

## Try It Out
✨ **Experience the magic of EGTTS V0.1 live!** Try the model directly through this [HuggingFace Space](https://huggingface.co/spaces/MohamedRashad/Egyptian-Arabic-TTS).

## Explore the Code
💻 **Dive into the implementation!** Check out the full code on [GitHub](https://github.com/joejoe03/Egyptian-Text-To-Speech).

## Quick Start
### Dependencies to install
```bash
pip install git+https://github.com/coqui-ai/TTS

pip install transformers

pip install deepspeed
```
### Inference
#### Load the model
```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CONFIG_FILE_PATH = 'path/to/config.json'
VOCAB_FILE_PATH = 'path/to/vocab.json'
MODEL_PATH = 'path/to/model'
SPEAKER_AUDIO_PATH = 'path/to/speaker.wav'

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_FILE_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, use_deepspeed=True, vocab_path=VOCAB_FILE_PATH)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_AUDIO_PATH])
```

#### Run the model
```python
from IPython.display import Audio, display

text = "صباح الخير"
print("Inference...")
out = model.inference(
    text,
    "ar",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.75,
)

AUDIO_OUTPUT_PATH = "path/to/output_audio.wav"
torchaudio.save("xtts_audio.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
display(Audio(AUDIO_OUTPUT_PATH, autoplay=True))
```

## Citation

```bibtex
@misc{OmarSamir,
      author = {Omar Samir, Youssef Waleed, Youssef Tamer ,and Amir Mohamed},
      title = {Fine-Tuning XTTS V2 for Egyptian Arabic},
      year = {2024},
      url = {https://github.com/joejoe03/Egyptian-Text-To-Speech},
}
```