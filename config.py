import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Image generation
SD_MODEL = "runwayml/stable-diffusion-v1-5"
SD_WIDTH, SD_HEIGHT = 768, 432
STYLE_PREFIX = (
    "modern anime style, cel-shaded, cinematic lighting, "
    "bokeh background, shallow depth of field, detailed eyes, "
    "realistic proportions, cool muted tones, dramatic mood, 2025 anime quality, "
)
NEGATIVE_PROMPT = (
    "chibi, flat shading, retro 90s anime, western cartoon, "
    "watercolor, low quality, blurry, deformed, ugly"
)
FIXED_SEED = 42
NUM_INFERENCE_STEPS = 25

FULL_STYLE_DESCRIPTION = (
    "modern high-quality anime style, cel-shaded, cinematic color grading, "
    "dramatic rim lighting, photorealistic bokeh background, shallow depth of field, "
    "large detailed eyes with multi-highlight irises, individual hair strand shading, "
    "realistic facial proportions, muted desaturated tones, cool blue-teal palette, "
    "melancholic emotional drama mood, extreme close-up portrait, "
    "2020s anime movie quality, volumetric bokeh lights, "
    "NOT chibi, NOT flat shading, NOT retro 90s anime, NOT western cartoon, NOT watercolor"
)

# LLM scene analysis
OLLAMA_MODEL = "llama3"
OLLAMA_HOST = "http://localhost:11434"
SEGMENT_METHOD = "sentence"

# TTS
TTS_VOICE = "en-US-AvaNeural"
TTS_RATE = "+0%"
TTS_VOLUME = "+0%"

# Video assembly
OUTPUT_FPS = 24
BGM_PATH = os.path.join(ASSETS_DIR, "bgm.mp3")
BGM_VOLUME = 0.3

# Ducking (FFmpeg sidechaincompress)
DUCK_THRESHOLD = 0.02
DUCK_RATIO = 4
DUCK_ATTACK = 5
DUCK_RELEASE = 200

# Ken Burns effect
KEN_BURNS_ZOOM_SPEED = 0.03

# Character consistency
CHARACTER_SHEET_PATH = os.path.join(OUTPUT_DIR, "characters.json")
