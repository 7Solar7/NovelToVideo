# Novel-to-Video

Automated pipeline that converts novel text into story narration videos with AI-generated visuals, TTS narration, subtitles, and background music.

## Pipeline Stages

1. **Scene Analysis** (Ollama + Llama 3) — splits text into sentences, groups them into visual segments, generates image prompts in a consistent anime style
2. **Character Extraction** (Ollama + Llama 3) — identifies named characters and their visual traits from the raw text; saves to `output/characters.json` for reuse across chapters
3. **TTS Narration + Subtitles** (edge-tts) — generates voiceover audio and sentence-level SRT subtitles
4. **Image Generation** (Stable Diffusion 1.5) — renders images for each segment with a fixed seed for visual consistency
5. **Video Assembly** (MoviePy + FFmpeg) — composites images with Ken Burns zoom effect, mixes narration with optional BGM ducking, burns subtitles

## Requirements

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (CUDA)
- FFmpeg (tested with gyan.dev essentials build 8.1.1)
- Ollama with Llama 3 (`ollama pull llama3`)

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

Generation of chapter video:
```bash
python main.py chapter1.txt output_video_name
```

Then use the character sheet from chapter 1 to maintain character appearance in chapter 2:
```bash
python main.py chapter2.txt chapter2_video --characters output/characters.json
```

## File Structure

```
novel-to-video/
├── main.py              # Pipeline orchestrator
├── config.py            # All constants (style, resolution, paths)
├── scene_analyzer.py    # Ollama-based prompt generation + character extraction
├── image_generator.py   # Stable Diffusion image generation
├── tts_generator.py     # edge-tts narration + subtitles
├── video_assembler.py   # MoviePy + FFmpeg video composition
├── requirements.txt     # Python dependencies
├── chapter1.txt         # Input novel text
├── assets/              # Place bgm.mp3 here for background music
└── output/              # Generated videos, images, audio, SRT, character sheets
```

## Configuration

All settings in `config.py`:
- **Style**: `STYLE_PREFIX` (positive prompt prefix), `NEGATIVE_PROMPT`, `FULL_STYLE_DESCRIPTION` (Ollama style instruction)
- **Resolution**: 768×432 (16:9)
- **TTS**: Microsoft edge-tts voice selection
- **Audio ducking**: FFmpeg sidechaincompress parameters
- **Ken Burns**: slow zoom speed

## Notes

- Images are cached in `output/` — delete them to regenerate with new settings
- The NOT-items in the style description are passed as SD's `negative_prompt` for better rejection of unwanted styles
- CLIP 77-token limit: style prefix (~31 tokens) + scene description (~46 tokens)
