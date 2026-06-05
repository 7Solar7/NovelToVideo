# Novel-to-Video

Automated pipeline that converts novel text into story narration videos with AI-generated anime-style visuals, TTS narration, subtitles, and background music.

## Pipeline Stages

1. **Scene Analysis** (Ollama + Llama 3) — splits text into sentences, groups them into visual segments, generates image prompts with a trigger word for LoRA consistency
2. **Character Extraction** (Ollama + Llama 3) — identifies named characters and their visual traits from the raw text; saves to `output/characters.json` for reuse across chapters
3. **TTS Narration + Subtitles** (edge-tts) — generates voiceover audio and sentence-level SRT subtitles (can run standalone, no GPU needed)
4. **Image Generation** (Stable Diffusion 1.5 + LoRA) — renders images using `hakurei/waifu-diffusion` with an optional Style LoRA fused for visual consistency
5. **Video Assembly** (MoviePy + FFmpeg) — composites images with Ken Burns zoom effect, mixes narration with optional BGM ducking, burns subtitles

## Requirements

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (CUDA) — required for image generation and LoRA training
- FFmpeg (tested with gyan.dev essentials build 8.1.1)
- Ollama with Llama 3 (`ollama pull llama3`)
- 8GB+ free disk for model cache (~4GB for waifu-diffusion)

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama and pull the model
# https://ollama.com
ollama pull llama3

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## Usage

### Full pipeline (image + audio + video)
```bash
python main.py chapter1.txt my_video
```

### Reuse character sheet for consistency across chapters
```bash
python main.py chapter2.txt ch2_video --characters output/characters.json
```

### Generate audio + subtitles only (no GPU needed)
```bash
python generate_audio.py chapter1.txt my_audio
python generate_audio.py chapter1.txt --voice en-US-JennyNeural --rate -10%
python generate_audio.py --list-voices    # List all available edge-tts voices
```

### Train a custom Style LoRA
```bash
# Place training images in training_images/ (at least 15-30 recommended)
python train_lora.py --steps 1000 --lr 1e-4

# The trained LoRA is auto-loaded by the pipeline if it exists at:
#   models/my_anime_style.safetensors
```

## File Structure

```
novel-to-video/
├── main.py                 # Pipeline orchestrator
├── config.py               # All constants (model, style, resolution, paths, LoRA)
├── scene_analyzer.py       # Ollama-based prompt generation + character extraction
├── image_generator.py      # Stable Diffusion + LoRA image generation
├── tts_generator.py        # edge-tts narration + subtitles
├── video_assembler.py      # MoviePy + FFmpeg video composition
├── generate_audio.py       # Standalone TTS + subtitles (no GPU)
├── train_lora.py           # LoRA training script (diffusers + peft)
├── requirements.txt        # Python dependencies
├── chapter1.txt            # Sample input novel text
├── training_images/        # Place reference images here for LoRA training
├── assets/                 # Place bgm.mp3 here for background music
├── models/                 # LoRA checkpoints and base model files
│   └── my_anime_style.safetensors  # Trained Style LoRA (auto-loaded)
└── output/                 # Generated videos, images, audio, SRT, character sheets
```

## Configuration

All settings in `config.py`:

| Field | Default | Description |
|---|---|---|
| `SD_MODEL` | `hakurei/waifu-diffusion` | Base Stable Diffusion model (diffusers format) |
| `STYLE_PREFIX` | — | Positive prompt prefix for all generated images |
| `NEGATIVE_PROMPT` | — | Negative prompt to reject unwanted styles |
| `FULL_STYLE_DESCRIPTION` | — | Style context sent to Ollama for prompt generation |
| `LORA_PATH` | `models/my_anime_style.safetensors` | Auto-loaded LoRA weights if file exists |
| `LORA_SCALE` | `0.8` | LoRA blend strength (0.0–1.0) |
| `TRIGGER_WORD` | `refanime_style` | Trigger word for LoRA activation in prompts |
| **Resolution** | 768×432 (16:9) | — |
| **TTS** | `en-US-AvaNeural` | Microsoft edge-tts voice |
| **Audio ducking** | threshold=0.02, ratio=4 | FFmpeg sidechaincompress parameters |
| **Ken Burns** | 3% zoom/second | Slow zoom speed |

## LoRA Training

To match a specific anime style, train a Style LoRA on reference images:

1. Place 15–120 reference PNG/JPG frames in `training_images/`
2. Run: `python train_lora.py --steps 1000`
3. The trained LoRA (`models/my_anime_style.safetensors`) is auto-loaded by the pipeline
4. The trigger word `refanime_style` is injected into all Ollama-generated prompts

Training at 512×512 with batch_size=1 fits in 4GB VRAM. Pre-computed VAE latents make training ~4x faster (~1000 steps in 1 hour on RTX 3050).

## Notes

- Images are cached in `output/` — delete them to regenerate with new settings or after training a new LoRA
- The `NOT` items in the style description are passed as SD's `negative_prompt` for better rejection of unwanted styles
- CLIP 77-token limit: style prefix (~31 tokens) + scene description (~46 tokens)
